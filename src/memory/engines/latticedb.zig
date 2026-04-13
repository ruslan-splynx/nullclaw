//! LatticeDB memory backend.
//!
//! Stores agent memory as a property graph:
//!   - Entry nodes         (label "Entry")        — props: key, created_at
//!   - ContentChunk nodes  (label "ContentChunk") — props: seq, data
//!   - Category nodes      (label "Category")     — props: name
//!   - Session nodes       (label "Session")      — props: id
//!   - HAS_CHUNK edges from each Entry to its ContentChunks
//!   - BELONGS_TO_CATEGORY edges from each Entry to its Category
//!   - IN_SESSION edges from each Entry to its Session (when session_id is set)
//!   - FTS index on Entry (via the full reconstructed content) for recall() via BM25
//!
//! Why content lives in child nodes: the btree layer underneath
//! LatticeDB stores values inline in a single leaf page, so no single
//! property value can exceed `lattice_open_options.page_size`. Splitting
//! content across ContentChunk child nodes (each ≤ CONTENT_CHUNK_SIZE
//! bytes of `data`) keeps every individual node well under the page
//! size while still letting a single Entry hold arbitrary-length
//! content. Upstream lattice `feat/expose-zig-module` (commits
//! `da98e02`, `b7eeb87`) previously had two additional hardcoded stack
//! buffers (`[512]u8` for WAL value payloads in database.zig and
//! `[4096]u8` for whole-node serialization in graph/node.zig) and a
//! panic inside `lattice_fts_index` on multi-KB docs; those are fixed
//! at source, so this adapter no longer caps FTS indexing at 2 KiB and
//! no longer needs to avoid inline STRING properties smaller than the
//! page size.
//!
//! v1 limitation: the in-memory key→node cache is not rebuilt from disk on
//! reopen. Fresh-DB workflows (contract tests, first-run agents) are unaffected;
//! persistent multi-process reopen is a follow-up once lattice exposes a
//! bulk-scan API we can call from the adapter.

const std = @import("std");
const Allocator = std.mem.Allocator;
const root = @import("../root.zig");
const Memory = root.Memory;
const MemoryCategory = root.MemoryCategory;
const MemoryEntry = root.MemoryEntry;
const log = std.log.scoped(.latticedb_memory);

const lattice = @import("lattice");
const c = lattice.c_api;

const NodeId = c.lattice_node_id;
const EdgeId = c.lattice_edge_id;

const Error = error{
    LatticeOpenFailed,
    LatticeTxnFailed,
    LatticeOpFailed,
    OutOfMemory,
};

fn mapCErr(code: c.lattice_error) Error!void {
    if (code == .ok) return;
    return switch (code) {
        .err_out_of_memory => Error.OutOfMemory,
        else => Error.LatticeOpFailed,
    };
}

const ENTRY_LABEL: [:0]const u8 = "Entry";
const CONTENT_CHUNK_LABEL: [:0]const u8 = "ContentChunk";
const CATEGORY_LABEL: [:0]const u8 = "Category";
const SESSION_LABEL: [:0]const u8 = "Session";
const HAS_CHUNK: [:0]const u8 = "HAS_CHUNK";
const BELONGS_TO_CATEGORY: [:0]const u8 = "BELONGS_TO_CATEGORY";
const IN_SESSION: [:0]const u8 = "IN_SESSION";
const PROP_KEY: [:0]const u8 = "key";
const PROP_CREATED_AT: [:0]const u8 = "created_at";
const PROP_NAME: [:0]const u8 = "name";
const PROP_ID: [:0]const u8 = "id";
const PROP_SEQ: [:0]const u8 = "seq";
const PROP_DATA: [:0]const u8 = "data";

/// Max payload bytes per ContentChunk.data string property. Kept well
/// below the observed ~507-byte per-STRING-property cap in LatticeDB
/// 0.5.0, and well below the ~4000-byte per-node-serialization cap
/// (each chunk node carries only `seq` + `data`, so total node size is
/// CONTENT_CHUNK_SIZE + ~30 bytes of framing).
const CONTENT_CHUNK_SIZE: usize = 400;

pub const LatticeMemory = struct {
    allocator: Allocator,
    db: *c.lattice_database,
    key_to_node: std.StringHashMapUnmanaged(NodeId) = .empty,
    category_to_node: std.StringHashMapUnmanaged(NodeId) = .empty,
    session_to_node: std.StringHashMapUnmanaged(NodeId) = .empty,
    owns_self: bool = false,

    const Self = @This();

    pub fn init(allocator: Allocator, path: []const u8) !Self {
        const path_z = try allocator.dupeZ(u8, path);
        defer allocator.free(path_z);

        var options = c.lattice_open_options{
            .create = true,
            .read_only = false,
            .cache_size_mb = 16,
            // Default 4 KiB pages cap a single STRING property at ~256 bytes
            // before the storage engine returns err_io. Onboard bootstrap
            // templates (SOUL.md, AGENTS.md …) run up to ~10 KiB, so bump
            // pages to 64 KiB which lifts the per-property ceiling above
            // anything the scaffold writes.
            .page_size = 65536,
            .enable_vector = false,
            .vector_dimensions = 0,
        };

        var db_ptr: ?*c.lattice_database = null;
        const open_err = c.lattice_open(path_z.ptr, &options, &db_ptr);
        if (open_err != .ok or db_ptr == null) {
            log.err("lattice_open failed for '{s}': {any}", .{ path, open_err });
            return Error.LatticeOpenFailed;
        }

        return Self{
            .allocator = allocator,
            .db = db_ptr.?,
        };
    }

    pub fn deinit(self: *Self) void {
        freeOwnedKeys(self.allocator, &self.key_to_node);
        freeOwnedKeys(self.allocator, &self.category_to_node);
        freeOwnedKeys(self.allocator, &self.session_to_node);
        _ = c.lattice_close(self.db);
        if (self.owns_self) self.allocator.destroy(self);
    }

    fn freeOwnedKeys(alloc: Allocator, map: *std.StringHashMapUnmanaged(NodeId)) void {
        var it = map.iterator();
        while (it.next()) |entry| {
            alloc.free(entry.key_ptr.*);
        }
        map.deinit(alloc);
    }

    pub fn memory(self: *Self) Memory {
        return .{
            .ptr = @ptrCast(self),
            .vtable = &vtable,
        };
    }

    // ── Transaction helpers ──────────────────────────────────────────

    fn beginWrite(self: *Self) !*c.lattice_txn {
        var txn: ?*c.lattice_txn = null;
        const err = c.lattice_begin(self.db, .read_write, &txn);
        if (err != .ok or txn == null) {
            log.err("lattice_begin write failed: {any}", .{err});
            return Error.LatticeTxnFailed;
        }
        return txn.?;
    }

    fn beginRead(self: *Self) !*c.lattice_txn {
        var txn: ?*c.lattice_txn = null;
        const err = c.lattice_begin(self.db, .read_only, &txn);
        if (err != .ok or txn == null) {
            log.err("lattice_begin read failed: {any}", .{err});
            return Error.LatticeTxnFailed;
        }
        return txn.?;
    }

    fn commit(txn: *c.lattice_txn) !void {
        const err = c.lattice_commit(txn);
        if (err != .ok) {
            log.err("lattice_commit failed: {any}", .{err});
            return Error.LatticeTxnFailed;
        }
    }

    fn rollback(txn: *c.lattice_txn) void {
        _ = c.lattice_rollback(txn);
    }

    // ── Value constructors ───────────────────────────────────────────

    fn stringValue(ptr: [*c]const u8, len: usize) c.lattice_value {
        return c.lattice_value{
            .value_type = .string,
            .data = .{ .string_val = .{ .ptr = ptr, .len = len } },
        };
    }

    fn intValue(v: i64) c.lattice_value {
        return c.lattice_value{
            .value_type = .int,
            .data = .{ .int_val = v },
        };
    }

    fn setNodePropertyString(
        txn: *c.lattice_txn,
        node_id: NodeId,
        key_z: [:0]const u8,
        value: []const u8,
    ) !void {
        var val = stringValue(value.ptr, value.len);
        try mapCErr(c.lattice_node_set_property(txn, node_id, key_z.ptr, &val));
    }

    /// Split `content` into CONTENT_CHUNK_SIZE-sized `ContentChunk` nodes
    /// and attach them to `entry_node_id` via HAS_CHUNK edges. Each chunk
    /// carries a `seq` int (0-based) and a `data` string so that readers
    /// can reassemble the content in order regardless of edge iteration
    /// order. Empty content creates zero chunk nodes.
    fn writeContentChunks(
        txn: *c.lattice_txn,
        entry_node_id: NodeId,
        content: []const u8,
    ) !void {
        if (content.len == 0) return;

        var offset: usize = 0;
        var seq: i64 = 0;
        while (offset < content.len) : (seq += 1) {
            const end = @min(offset + CONTENT_CHUNK_SIZE, content.len);
            const slice = content[offset..end];

            var chunk_id: NodeId = 0;
            try mapCErr(c.lattice_node_create(txn, CONTENT_CHUNK_LABEL.ptr, &chunk_id));
            try setNodePropertyInt(txn, chunk_id, PROP_SEQ, seq);
            try setNodePropertyString(txn, chunk_id, PROP_DATA, slice);

            var edge_id: EdgeId = 0;
            try mapCErr(c.lattice_edge_create(txn, entry_node_id, chunk_id, HAS_CHUNK.ptr, &edge_id));

            offset = end;
        }
    }

    /// Walk the outgoing HAS_CHUNK edges of `entry_node_id`, read each
    /// chunk's (seq, data), sort by seq, and return a freshly-allocated
    /// contiguous slice. An Entry with zero HAS_CHUNK edges is treated as
    /// holding empty content (returns a zero-length owned slice), so
    /// loadEntry can distinguish "missing Entry" from "empty Entry" via
    /// the presence of the Entry's own key/created_at properties.
    fn readContentChunks(
        allocator: Allocator,
        txn: *c.lattice_txn,
        entry_node_id: NodeId,
    ) ![]u8 {
        var edges: ?*c.lattice_edge_result = null;
        try mapCErr(c.lattice_edge_get_outgoing(txn, entry_node_id, &edges));
        if (edges == null) return try allocator.alloc(u8, 0);
        defer c.lattice_edge_result_free(edges);

        const edge_count = c.lattice_edge_result_count(edges);

        const ChunkPiece = struct {
            seq: i64,
            data: []u8,
        };

        var pieces: std.ArrayListUnmanaged(ChunkPiece) = .empty;
        defer {
            for (pieces.items) |p| allocator.free(p.data);
            pieces.deinit(allocator);
        }

        var i: u32 = 0;
        while (i < edge_count) : (i += 1) {
            var source: NodeId = 0;
            var target: NodeId = 0;
            var type_ptr: [*c]const u8 = undefined;
            var type_len: c_uint = 0;
            try mapCErr(c.lattice_edge_result_get(edges, i, &source, &target, &type_ptr, &type_len));

            const etype = type_ptr[0..@as(usize, type_len)];
            if (!std.mem.eql(u8, etype, HAS_CHUNK)) continue;

            const seq = (try readIntProperty(txn, target, PROP_SEQ)) orelse continue;
            const data = (try readStringProperty(allocator, txn, target, PROP_DATA)) orelse continue;
            errdefer allocator.free(data);

            try pieces.append(allocator, .{ .seq = seq, .data = data });
        }

        std.mem.sort(ChunkPiece, pieces.items, {}, struct {
            fn lt(_: void, a: ChunkPiece, b: ChunkPiece) bool {
                return a.seq < b.seq;
            }
        }.lt);

        var total: usize = 0;
        for (pieces.items) |p| total += p.data.len;

        const out = try allocator.alloc(u8, total);
        var cursor: usize = 0;
        for (pieces.items) |p| {
            @memcpy(out[cursor .. cursor + p.data.len], p.data);
            cursor += p.data.len;
        }
        return out;
    }

    /// Walk outgoing HAS_CHUNK edges and return the list of target node
    /// IDs so `implForget` can cascade-delete them before dropping the
    /// parent Entry. Caller owns the returned slice.
    fn collectChunkNodeIds(
        allocator: Allocator,
        txn: *c.lattice_txn,
        entry_node_id: NodeId,
    ) ![]NodeId {
        var edges: ?*c.lattice_edge_result = null;
        try mapCErr(c.lattice_edge_get_outgoing(txn, entry_node_id, &edges));
        if (edges == null) return try allocator.alloc(NodeId, 0);
        defer c.lattice_edge_result_free(edges);

        var out: std.ArrayListUnmanaged(NodeId) = .empty;
        errdefer out.deinit(allocator);

        const count = c.lattice_edge_result_count(edges);
        var i: u32 = 0;
        while (i < count) : (i += 1) {
            var source: NodeId = 0;
            var target: NodeId = 0;
            var type_ptr: [*c]const u8 = undefined;
            var type_len: c_uint = 0;
            try mapCErr(c.lattice_edge_result_get(edges, i, &source, &target, &type_ptr, &type_len));

            const etype = type_ptr[0..@as(usize, type_len)];
            if (std.mem.eql(u8, etype, HAS_CHUNK)) {
                try out.append(allocator, target);
            }
        }

        return try out.toOwnedSlice(allocator);
    }

    fn setNodePropertyInt(
        txn: *c.lattice_txn,
        node_id: NodeId,
        key_z: [:0]const u8,
        v: i64,
    ) !void {
        var val = intValue(v);
        try mapCErr(c.lattice_node_set_property(txn, node_id, key_z.ptr, &val));
    }

    fn readStringProperty(
        allocator: Allocator,
        txn: *c.lattice_txn,
        node_id: NodeId,
        key_z: [:0]const u8,
    ) !?[]u8 {
        var out: c.lattice_value = std.mem.zeroes(c.lattice_value);
        const err = c.lattice_node_get_property(txn, node_id, key_z.ptr, &out);
        if (err == .err_not_found) return null;
        try mapCErr(err);
        defer c.lattice_value_free(&out);

        if (out.value_type != .string) return null;
        const str = out.data.string_val;
        if (str.len == 0) return try allocator.dupe(u8, "");
        return try allocator.dupe(u8, str.ptr[0..str.len]);
    }

    fn readIntProperty(
        txn: *c.lattice_txn,
        node_id: NodeId,
        key_z: [:0]const u8,
    ) !?i64 {
        var out: c.lattice_value = std.mem.zeroes(c.lattice_value);
        const err = c.lattice_node_get_property(txn, node_id, key_z.ptr, &out);
        if (err == .err_not_found) return null;
        try mapCErr(err);
        defer c.lattice_value_free(&out);

        if (out.value_type != .int) return null;
        return out.data.int_val;
    }

    // ── Cache-aware category/session node lookup/create ──────────────

    fn getOrCreateCategoryNode(
        self: *Self,
        txn: *c.lattice_txn,
        name: []const u8,
    ) !NodeId {
        if (self.category_to_node.get(name)) |id| return id;

        var node_id: NodeId = 0;
        try mapCErr(c.lattice_node_create(txn, CATEGORY_LABEL.ptr, &node_id));
        try setNodePropertyString(txn, node_id, PROP_NAME, name);

        const owned = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned);
        try self.category_to_node.put(self.allocator, owned, node_id);
        return node_id;
    }

    fn getOrCreateSessionNode(
        self: *Self,
        txn: *c.lattice_txn,
        session_id: []const u8,
    ) !NodeId {
        if (self.session_to_node.get(session_id)) |id| return id;

        var node_id: NodeId = 0;
        try mapCErr(c.lattice_node_create(txn, SESSION_LABEL.ptr, &node_id));
        try setNodePropertyString(txn, node_id, PROP_ID, session_id);

        const owned = try self.allocator.dupe(u8, session_id);
        errdefer self.allocator.free(owned);
        try self.session_to_node.put(self.allocator, owned, node_id);
        return node_id;
    }

    // ── Edge traversal readers ───────────────────────────────────────

    /// Walk the outgoing edges of `entry_node_id` looking for a BELONGS_TO_CATEGORY
    /// target, then read the target's `name` property. Caller owns the returned slice.
    fn readEntryCategory(
        allocator: Allocator,
        txn: *c.lattice_txn,
        entry_node_id: NodeId,
    ) !?[]u8 {
        var edges: ?*c.lattice_edge_result = null;
        try mapCErr(c.lattice_edge_get_outgoing(txn, entry_node_id, &edges));
        if (edges == null) return null;
        defer c.lattice_edge_result_free(edges);

        const count = c.lattice_edge_result_count(edges);
        var i: u32 = 0;
        while (i < count) : (i += 1) {
            var source: NodeId = 0;
            var target: NodeId = 0;
            var type_ptr: [*c]const u8 = undefined;
            var type_len: c_uint = 0;
            try mapCErr(c.lattice_edge_result_get(edges, i, &source, &target, &type_ptr, &type_len));

            const etype = type_ptr[0..@as(usize, type_len)];
            if (std.mem.eql(u8, etype, BELONGS_TO_CATEGORY)) {
                return try readStringProperty(allocator, txn, target, PROP_NAME);
            }
        }
        return null;
    }

    /// Walk the outgoing edges of `entry_node_id` looking for an IN_SESSION target,
    /// then read the target's `id` property. Caller owns the returned slice.
    fn readEntrySession(
        allocator: Allocator,
        txn: *c.lattice_txn,
        entry_node_id: NodeId,
    ) !?[]u8 {
        var edges: ?*c.lattice_edge_result = null;
        try mapCErr(c.lattice_edge_get_outgoing(txn, entry_node_id, &edges));
        if (edges == null) return null;
        defer c.lattice_edge_result_free(edges);

        const count = c.lattice_edge_result_count(edges);
        var i: u32 = 0;
        while (i < count) : (i += 1) {
            var source: NodeId = 0;
            var target: NodeId = 0;
            var type_ptr: [*c]const u8 = undefined;
            var type_len: c_uint = 0;
            try mapCErr(c.lattice_edge_result_get(edges, i, &source, &target, &type_ptr, &type_len));

            const etype = type_ptr[0..@as(usize, type_len)];
            if (std.mem.eql(u8, etype, IN_SESSION)) {
                return try readStringProperty(allocator, txn, target, PROP_ID);
            }
        }
        return null;
    }

    /// Build a MemoryEntry from an entry node id.
    /// All returned strings are freshly allocated via `allocator`.
    fn loadEntry(
        allocator: Allocator,
        txn: *c.lattice_txn,
        entry_node_id: NodeId,
    ) !?MemoryEntry {
        const key = (try readStringProperty(allocator, txn, entry_node_id, PROP_KEY)) orelse return null;
        errdefer allocator.free(key);

        const content = try readContentChunks(allocator, txn, entry_node_id);
        errdefer allocator.free(content);

        const created_at_ms = (try readIntProperty(txn, entry_node_id, PROP_CREATED_AT)) orelse 0;
        const timestamp = try std.fmt.allocPrint(allocator, "{d}", .{created_at_ms});
        errdefer allocator.free(timestamp);

        const category_name = (try readEntryCategory(allocator, txn, entry_node_id)) orelse
            try allocator.dupe(u8, "core");
        // category_name is either consumed into a .custom MemoryCategory (transfer of
        // ownership) or freed in one of the known-category branches below.

        const category: MemoryCategory = blk: {
            if (std.mem.eql(u8, category_name, "core")) {
                allocator.free(category_name);
                break :blk .core;
            }
            if (std.mem.eql(u8, category_name, "daily")) {
                allocator.free(category_name);
                break :blk .daily;
            }
            if (std.mem.eql(u8, category_name, "conversation")) {
                allocator.free(category_name);
                break :blk .conversation;
            }
            break :blk .{ .custom = category_name };
        };
        errdefer switch (category) {
            .custom => |name| allocator.free(name),
            else => {},
        };

        const session_id_opt = try readEntrySession(allocator, txn, entry_node_id);
        errdefer if (session_id_opt) |sid| allocator.free(sid);

        const id_str = try std.fmt.allocPrint(allocator, "{d}", .{entry_node_id});

        return MemoryEntry{
            .id = id_str,
            .key = key,
            .content = content,
            .category = category,
            .timestamp = timestamp,
            .session_id = session_id_opt,
            .score = null,
        };
    }

    // ── VTable impls ─────────────────────────────────────────────────

    fn implName(_: *anyopaque) []const u8 {
        return "latticedb";
    }

    fn implStore(
        ptr: *anyopaque,
        key: []const u8,
        content: []const u8,
        category: MemoryCategory,
        session_id: ?[]const u8,
    ) anyerror!void {
        const self: *Self = @ptrCast(@alignCast(ptr));

        // Overwrite semantics: delete any existing entry with the same key first.
        // implForget commits its own transaction before we begin the insert txn.
        if (self.key_to_node.get(key) != null) {
            _ = try implForget(ptr, key);
        }

        // Pre-allocate the cache-key dupe so an OOM surfaces before we commit.
        const owned_key = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(owned_key);

        const txn = try self.beginWrite();
        var committed = false;
        errdefer if (!committed) rollback(txn);

        var entry_id: NodeId = 0;
        try mapCErr(c.lattice_node_create(txn, ENTRY_LABEL.ptr, &entry_id));

        try setNodePropertyString(txn, entry_id, PROP_KEY, key);
        try setNodePropertyInt(txn, entry_id, PROP_CREATED_AT, std.time.milliTimestamp());
        try writeContentChunks(txn, entry_id, content);

        const cat_node = try self.getOrCreateCategoryNode(txn, category.toString());
        var cat_edge: EdgeId = 0;
        try mapCErr(c.lattice_edge_create(txn, entry_id, cat_node, BELONGS_TO_CATEGORY.ptr, &cat_edge));

        if (session_id) |sid| {
            const sess_node = try self.getOrCreateSessionNode(txn, sid);
            var sess_edge: EdgeId = 0;
            try mapCErr(c.lattice_edge_create(txn, entry_id, sess_node, IN_SESSION.ptr, &sess_edge));
        }

        if (content.len > 0) {
            // lattice 0.5.1+ (feat/expose-zig-module: da98e02, b7eeb87)
            // tokenizes the full content and gracefully degrades the
            // per-doc reverse-index when it would exceed a single btree
            // leaf page, so we can hand over the entire content without
            // the 2 KiB cap the earlier nullclaw workaround imposed.
            try mapCErr(c.lattice_fts_index(txn, entry_id, content.ptr, content.len));
        }

        try commit(txn);
        committed = true;

        try self.key_to_node.put(self.allocator, owned_key, entry_id);
    }

    fn implRecall(
        ptr: *anyopaque,
        allocator: Allocator,
        query: []const u8,
        limit: usize,
        session_id: ?[]const u8,
    ) anyerror![]MemoryEntry {
        const self: *Self = @ptrCast(@alignCast(ptr));

        if (query.len == 0 or limit == 0) {
            return try allocator.alloc(MemoryEntry, 0);
        }

        const search_limit: u32 = @intCast(@min(limit * 4, @as(usize, std.math.maxInt(u32))));
        var results: ?*c.lattice_fts_result = null;
        const err = c.lattice_fts_search(self.db, query.ptr, query.len, search_limit, &results);
        if (err != .ok or results == null) {
            return try allocator.alloc(MemoryEntry, 0);
        }
        defer c.lattice_fts_result_free(results);

        const count = c.lattice_fts_result_count(results);
        if (count == 0) return try allocator.alloc(MemoryEntry, 0);

        const txn = try self.beginRead();
        defer rollback(txn);

        var out: std.ArrayListUnmanaged(MemoryEntry) = .empty;
        errdefer {
            for (out.items) |*e| e.deinit(allocator);
            out.deinit(allocator);
        }

        var i: u32 = 0;
        while (i < count and out.items.len < limit) : (i += 1) {
            var node_id: NodeId = 0;
            var score: f32 = 0;
            if (c.lattice_fts_result_get(results, i, &node_id, &score) != .ok) continue;

            var entry = (try loadEntry(allocator, txn, node_id)) orelse continue;

            if (session_id) |want_sid| {
                const match = if (entry.session_id) |got| std.mem.eql(u8, got, want_sid) else false;
                if (!match) {
                    entry.deinit(allocator);
                    continue;
                }
            }

            entry.score = @floatCast(score);
            try out.append(allocator, entry);
        }

        return try out.toOwnedSlice(allocator);
    }

    fn implGet(
        ptr: *anyopaque,
        allocator: Allocator,
        key: []const u8,
    ) anyerror!?MemoryEntry {
        const self: *Self = @ptrCast(@alignCast(ptr));
        const node_id = self.key_to_node.get(key) orelse return null;

        const txn = try self.beginRead();
        defer rollback(txn);

        return try loadEntry(allocator, txn, node_id);
    }

    fn implList(
        ptr: *anyopaque,
        allocator: Allocator,
        category: ?MemoryCategory,
        session_id: ?[]const u8,
    ) anyerror![]MemoryEntry {
        const self: *Self = @ptrCast(@alignCast(ptr));

        const txn = try self.beginRead();
        defer rollback(txn);

        var out: std.ArrayListUnmanaged(MemoryEntry) = .empty;
        errdefer {
            for (out.items) |*e| e.deinit(allocator);
            out.deinit(allocator);
        }

        var it = self.key_to_node.iterator();
        while (it.next()) |kv| {
            const node_id = kv.value_ptr.*;
            var entry = (try loadEntry(allocator, txn, node_id)) orelse continue;

            if (category) |want_cat| {
                if (!entry.category.eql(want_cat)) {
                    entry.deinit(allocator);
                    continue;
                }
            }
            if (session_id) |want_sid| {
                const match = if (entry.session_id) |got| std.mem.eql(u8, got, want_sid) else false;
                if (!match) {
                    entry.deinit(allocator);
                    continue;
                }
            }

            try out.append(allocator, entry);
        }

        return try out.toOwnedSlice(allocator);
    }

    fn implForget(ptr: *anyopaque, key: []const u8) anyerror!bool {
        const self: *Self = @ptrCast(@alignCast(ptr));
        const node_id = self.key_to_node.get(key) orelse return false;

        const txn = try self.beginWrite();
        var committed = false;
        errdefer if (!committed) rollback(txn);

        // Collect HAS_CHUNK targets up-front — once we start deleting edges
        // the outgoing-edge iterator is unsafe to reuse.
        const chunk_ids = try collectChunkNodeIds(self.allocator, txn, node_id);
        defer self.allocator.free(chunk_ids);

        // Explicitly remove outgoing edges so the node delete never references
        // stale adjacency, regardless of whether lattice cascades.
        var edges: ?*c.lattice_edge_result = null;
        const ge = c.lattice_edge_get_outgoing(txn, node_id, &edges);
        if (ge == .ok and edges != null) {
            defer c.lattice_edge_result_free(edges);
            const count = c.lattice_edge_result_count(edges);
            var i: u32 = 0;
            while (i < count) : (i += 1) {
                var source: NodeId = 0;
                var target: NodeId = 0;
                var type_ptr: [*c]const u8 = undefined;
                var type_len: c_uint = 0;
                if (c.lattice_edge_result_get(edges, i, &source, &target, &type_ptr, &type_len) != .ok) continue;

                const etype_slice = type_ptr[0..@as(usize, type_len)];
                const etype_z = try self.allocator.dupeZ(u8, etype_slice);
                defer self.allocator.free(etype_z);
                _ = c.lattice_edge_delete(txn, source, target, etype_z.ptr);
            }
        }

        // Drop the orphaned ContentChunk nodes that the Entry used to own.
        // Best-effort: if a chunk node is already gone (race with another
        // writer on reopen), lattice returns not_found which we ignore.
        for (chunk_ids) |cid| {
            _ = c.lattice_node_delete(txn, cid);
        }

        try mapCErr(c.lattice_node_delete(txn, node_id));
        try commit(txn);
        committed = true;

        if (self.key_to_node.fetchRemove(key)) |kv| {
            self.allocator.free(kv.key);
        }
        return true;
    }

    fn implCount(ptr: *anyopaque) anyerror!usize {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.key_to_node.count();
    }

    fn implHealthCheck(_: *anyopaque) bool {
        return true;
    }

    fn implDeinit(ptr: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.deinit();
    }

    const vtable = Memory.VTable{
        .name = &implName,
        .store = &implStore,
        .recall = &implRecall,
        .get = &implGet,
        .list = &implList,
        .forget = &implForget,
        .count = &implCount,
        .healthCheck = &implHealthCheck,
        .deinit = &implDeinit,
    };
};

// ── Tests ──────────────────────────────────────────────────────────

fn openTmp(tmp_dir: *std.testing.TmpDir) !struct {
    path: [:0]u8,
    base: []u8,
} {
    const base = try tmp_dir.dir.realpathAlloc(std.testing.allocator, ".");
    errdefer std.testing.allocator.free(base);
    const path = try std.fs.path.joinZ(std.testing.allocator, &.{ base, "lattice.db" });
    return .{ .path = path, .base = base };
}

test "latticedb memory smoke" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const paths = try openTmp(&tmp);
    defer std.testing.allocator.free(paths.path);
    defer std.testing.allocator.free(paths.base);

    var mem = try LatticeMemory.init(std.testing.allocator, paths.path);
    defer mem.deinit();
    const m = mem.memory();

    try std.testing.expectEqualStrings("latticedb", m.name());
    try std.testing.expect(m.healthCheck());
    try std.testing.expectEqual(@as(usize, 0), try m.count());

    try m.store("k1", "hello world", .core, null);
    try std.testing.expectEqual(@as(usize, 1), try m.count());

    {
        const got = try m.get(std.testing.allocator, "k1");
        try std.testing.expect(got != null);
        defer got.?.deinit(std.testing.allocator);
        try std.testing.expectEqualStrings("k1", got.?.key);
        try std.testing.expectEqualStrings("hello world", got.?.content);
        try std.testing.expect(got.?.category.eql(.core));
    }

    {
        const recalled = try m.recall(std.testing.allocator, "hello", 10, null);
        defer root.freeEntries(std.testing.allocator, recalled);
        try std.testing.expect(recalled.len >= 1);
    }

    const forgotten = try m.forget("k1");
    try std.testing.expect(forgotten);
    try std.testing.expectEqual(@as(usize, 0), try m.count());
}

test "latticedb memory round-trips multi-KB content via chunk nodes" {
    // Regression guard for the LatticeDB 0.5.0 size caps that are now
    // fixed upstream on feat/expose-zig-module (da98e02, b7eeb87):
    //   - ~507-byte per-STRING-property cap (former database.zig:1514 buf[512])
    //   - 4096-byte per-node serialization cap (former node.zig:199 buf[4096])
    //   - integer-overflow panic in lattice_fts_index above ~2 KiB
    // writeContentChunks splits content across child ContentChunk nodes
    // so each individual chunk stays under the btree leaf page size,
    // which lets the engine store arbitrary-length content regardless of
    // `lattice_open_options.page_size`. Covers the workspace-template
    // flow that `nullclaw onboard --memory latticedb` exercises.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const paths = try openTmp(&tmp);
    defer std.testing.allocator.free(paths.path);
    defer std.testing.allocator.free(paths.base);

    var mem = try LatticeMemory.init(std.testing.allocator, paths.path);
    defer mem.deinit();
    const m = mem.memory();

    const sizes = [_]usize{ 64, 256, 512, 1024, 1700, 4000, 9242, 16384 };
    for (sizes) |sz| {
        const buf = try std.testing.allocator.alloc(u8, sz);
        defer std.testing.allocator.free(buf);
        // Distinct byte pattern per offset so we can verify ordering of
        // reassembled chunks, not just length.
        for (buf, 0..) |*b, i| b.* = @truncate(i);
        const key_buf = try std.fmt.allocPrint(std.testing.allocator, "k{d}", .{sz});
        defer std.testing.allocator.free(key_buf);

        try m.store(key_buf, buf, .core, null);

        const got = try m.get(std.testing.allocator, key_buf);
        try std.testing.expect(got != null);
        defer got.?.deinit(std.testing.allocator);
        try std.testing.expectEqual(@as(usize, sz), got.?.content.len);
        try std.testing.expectEqualSlices(u8, buf, got.?.content);
    }

    try std.testing.expectEqual(@as(usize, sizes.len), try m.count());
}

test "latticedb memory stores many entries with shared category" {
    // Reproduces the onboard scaffoldWorkspace flow: 7+ store calls sharing
    // category .core and no session_id. Earlier manual testing showed
    // `nullclaw onboard --memory latticedb` failing with LatticeOpFailed
    // once scaffold tried to write a run of bootstrap files.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const paths = try openTmp(&tmp);
    defer std.testing.allocator.free(paths.path);
    defer std.testing.allocator.free(paths.base);

    var mem = try LatticeMemory.init(std.testing.allocator, paths.path);
    defer mem.deinit();
    const m = mem.memory();

    const files = [_][]const u8{
        "SOUL.md",     "AGENTS.md", "TOOLS.md",     "CONFIG.md",
        "IDENTITY.md", "USER.md",   "HEARTBEAT.md", "BOOTSTRAP.md",
    };
    for (files) |f| {
        try m.store(f, "test content for bootstrap scaffold", .core, null);
    }
    try std.testing.expectEqual(@as(usize, files.len), try m.count());
}
