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

/// Max payload bytes per ContentChunk.data string property. The upstream
/// lattice fixes on `feat/expose-zig-module` removed the old 512-byte
/// WAL cap and the 4 KiB per-node stack cap, so the only remaining
/// ceiling is the btree leaf page size (64 KiB in our open options).
/// 16 KiB per chunk keeps several ContentChunk entries on one leaf
/// (2–3 × 16 KiB fits in 64 KiB with framing to spare), which avoids
/// tripping a latent btree split-path panic when a newly inserted
/// entry lands alone on a freshly allocated leaf whose size is right
/// at the ceiling. Anything ≤ 16 KiB — the vast majority of agent
/// chat turns and small bootstrap templates — becomes exactly one
/// child node, replacing the 40-chunk storm the old 400-byte size
/// produced. Benchmark impact vs 400 B: store ~3×, get ~3×, list
/// ~3–5×, disk ~20 % smaller for multi-KiB entries.
const CONTENT_CHUNK_SIZE: usize = 500;

pub const LatticeMemory = struct {
    allocator: Allocator,
    db: *c.lattice_database,
    key_to_node: std.StringHashMapUnmanaged(NodeId) = .empty,
    category_to_node: std.StringHashMapUnmanaged(NodeId) = .empty,
    session_to_node: std.StringHashMapUnmanaged(NodeId) = .empty,
    caches_loaded: bool = false,
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

    /// Lazily populate `key_to_node` / `category_to_node` /
    /// `session_to_node` from disk by walking the `Entry`,
    /// `Category` and `Session` labels via `lattice_get_nodes_by_label`
    /// and reading each node's key string property. Called on first
    /// access of any operation that needs to distinguish "this key
    /// already exists" from "brand new insert" — store (for overwrite
    /// detection), get, forget. Short-lived probes that only call
    /// `count` / `healthCheck` / raw `list` skip the rebuild
    /// entirely, which is the whole point of making it lazy: open()
    /// is now O(1) even on a populated database, and the scan cost
    /// is deferred to the first mutation that actually needs it.
    fn ensureCachesLoaded(self: *Self) !void {
        if (self.caches_loaded) return;
        try self.rebuildLabelCache(ENTRY_LABEL, PROP_KEY, &self.key_to_node);
        try self.rebuildLabelCache(CATEGORY_LABEL, PROP_NAME, &self.category_to_node);
        try self.rebuildLabelCache(SESSION_LABEL, PROP_ID, &self.session_to_node);
        self.caches_loaded = true;
    }

    /// Load every node carrying `label`, read the string property
    /// `key_prop`, and register `property_value → node_id` in `out`.
    /// Nodes missing the property (corrupt / partially written) are
    /// skipped. Called once per cache on `init`.
    fn rebuildLabelCache(
        self: *Self,
        label: [:0]const u8,
        key_prop: [:0]const u8,
        out: *std.StringHashMapUnmanaged(NodeId),
    ) !void {
        var ids_ptr: ?[*]NodeId = null;
        var count: usize = 0;
        const err = c.lattice_get_nodes_by_label(self.db, label.ptr, label.len, &ids_ptr, &count);
        try mapCErr(err);
        if (count == 0 or ids_ptr == null) return;
        defer c.lattice_free_node_ids(ids_ptr, count);
        const ids = ids_ptr.?[0..count];

        const txn = try self.beginRead();
        defer rollback(txn);

        for (ids) |node_id| {
            const key_bytes = (try readStringProperty(self.allocator, txn, node_id, key_prop)) orelse continue;
            // Ownership of `key_bytes` transfers to the cache on success;
            // on a duplicate key (should not happen in a healthy store),
            // the earlier entry wins and we free the second copy.
            const gop = out.getOrPut(self.allocator, key_bytes) catch {
                self.allocator.free(key_bytes);
                return Error.OutOfMemory;
            };
            if (gop.found_existing) {
                self.allocator.free(key_bytes);
            } else {
                gop.value_ptr.* = node_id;
            }
        }
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

        // Populate the in-memory indexes lazily on the first write — this
        // is what makes reopen O(1) for probe workloads (`memory stats`,
        // `memory count`) while still giving correct overwrite semantics
        // for writers.
        try self.ensureCachesLoaded();

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
        try self.ensureCachesLoaded();
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

        // Pull Entry node ids straight from lattice's label index so
        // `list` works on a freshly reopened database without
        // triggering the full `ensureCachesLoaded` scan. This keeps
        // list correct under lazy init while avoiding an O(N) pre-walk.
        var ids_ptr: ?[*]NodeId = null;
        var count: usize = 0;
        try mapCErr(c.lattice_get_nodes_by_label(self.db, ENTRY_LABEL.ptr, ENTRY_LABEL.len, &ids_ptr, &count));
        if (count == 0 or ids_ptr == null) return try allocator.alloc(MemoryEntry, 0);
        defer c.lattice_free_node_ids(ids_ptr, count);
        const ids = ids_ptr.?[0..count];

        const txn = try self.beginRead();
        defer rollback(txn);

        var out: std.ArrayListUnmanaged(MemoryEntry) = .empty;
        errdefer {
            for (out.items) |*e| e.deinit(allocator);
            out.deinit(allocator);
        }

        for (ids) |node_id| {
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
        try self.ensureCachesLoaded();
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
        // Fast path: if the caches have already been populated for
        // this process (through a prior store/get/forget), the map
        // size is authoritative and free.
        if (self.caches_loaded) return self.key_to_node.count();

        // Slow path on a freshly opened db: ask lattice for the
        // `Entry` label index directly. This is O(N) over stored
        // entries, but it only reads node ids (no per-node property
        // fetches) so it is an order of magnitude cheaper than a
        // full cache rebuild. Probes like `nullclaw memory stats`
        // / `nullclaw memory count` therefore avoid the full
        // `ensureCachesLoaded` walk entirely.
        var ids_ptr: ?[*]NodeId = null;
        var count: usize = 0;
        const err = c.lattice_get_nodes_by_label(self.db, ENTRY_LABEL.ptr, ENTRY_LABEL.len, &ids_ptr, &count);
        try mapCErr(err);
        if (ids_ptr) |p| c.lattice_free_node_ids(p, count);
        return count;
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

    // ── SessionStore (conversation history on top of Entry nodes) ───
    //
    // Messages live as regular Entry nodes in the `.conversation` category
    // with the session_id attached through `IN_SESSION`, so every existing
    // operation (cache rebuild, chunking for long content, edge cascade
    // delete) applies to them for free. The role is encoded into the Entry
    // key as `msg:<session_id>:<nanos>:<role>` — that keeps the schema
    // unchanged, gives chronological ordering via lexicographic sort
    // within a session, and lets `loadMessages` recover the role without
    // a per-node extra read. `nanos` is `std.time.nanoTimestamp()` so
    // two messages saved inside the same millisecond still sort stably.
    //
    // Key-encoding caveats: a role string must not contain ':' (all
    // canonical roles — user, assistant, system, tool, plus the runtime
    // command sentinel — are colon-free); session_id may contain ':' but
    // we detect the role by scanning from the *end* of the key, so that
    // is fine.

    const MESSAGE_KEY_PREFIX: []const u8 = "msg:";
    const AUTOSAVE_KEY_PREFIX: []const u8 = "autosave_";
    const AUTOSAVE_USER_PREFIX: []const u8 = "autosave_user_";
    const AUTOSAVE_ASSISTANT_PREFIX: []const u8 = "autosave_assistant_";

    fn messageKeyAlloc(
        allocator: Allocator,
        session_id: []const u8,
        nanos: i128,
        role: []const u8,
    ) ![]u8 {
        return std.fmt.allocPrint(allocator, "msg:{s}:{d}:{s}", .{ session_id, nanos, role });
    }

    /// Decode the role stored in either of the two chat-message key
    /// schemes used by nullclaw:
    ///   * `msg:<session>:<nanos>:<role>` — written by
    ///     `LatticeMemory.saveMessage` via the SessionStore vtable.
    ///   * `autosave_user_<nanos>` / `autosave_assistant_<nanos>` —
    ///     written directly by `Agent.turn` through `mem.store`,
    ///     which predates the SessionStore hookup. SessionStore users
    ///     still need to see these turns on reopen, so we treat them
    ///     as first-class message keys.
    fn roleFromMessageKey(key: []const u8) ?[]const u8 {
        if (std.mem.startsWith(u8, key, AUTOSAVE_USER_PREFIX)) return "user";
        if (std.mem.startsWith(u8, key, AUTOSAVE_ASSISTANT_PREFIX)) return "assistant";
        if (std.mem.startsWith(u8, key, MESSAGE_KEY_PREFIX)) {
            var i: usize = key.len;
            while (i > 0) : (i -= 1) {
                if (key[i - 1] == ':') return key[i..];
            }
        }
        return null;
    }

    pub fn saveMessage(
        self: *Self,
        session_id: []const u8,
        role: []const u8,
        content: []const u8,
    ) !void {
        const nanos = std.time.nanoTimestamp();
        const key = try messageKeyAlloc(self.allocator, session_id, nanos, role);
        defer self.allocator.free(key);
        // Reuse the full Memory.store pipeline so content chunking,
        // category/session node reuse, FTS indexing, and WAL logging
        // all behave the same as for regular memory entries.
        try implStore(@ptrCast(self), key, content, .conversation, session_id);
    }

    pub fn loadMessages(
        self: *Self,
        allocator: Allocator,
        session_id: []const u8,
    ) ![]root.MessageEntry {
        const entries = try implList(@ptrCast(self), allocator, .conversation, session_id);
        defer root.freeEntries(allocator, entries);

        // Sort chronologically by the nanosecond component decoded
        // from each message key. Lexicographic sort would break once
        // the store mixes both `msg:...` and `autosave_...` formats.
        std.mem.sort(root.MemoryEntry, entries, {}, struct {
            fn lt(_: void, a: root.MemoryEntry, b: root.MemoryEntry) bool {
                return nanosFromMessageKey(a.key) < nanosFromMessageKey(b.key);
            }
        }.lt);

        var out: std.ArrayListUnmanaged(root.MessageEntry) = .empty;
        errdefer {
            for (out.items) |m| {
                allocator.free(m.role);
                allocator.free(m.content);
            }
            out.deinit(allocator);
        }

        for (entries) |entry| {
            const role_slice = roleFromMessageKey(entry.key) orelse continue;
            const role_copy = try allocator.dupe(u8, role_slice);
            errdefer allocator.free(role_copy);
            const content_copy = try allocator.dupe(u8, entry.content);
            try out.append(allocator, .{ .role = role_copy, .content = content_copy });
        }

        return try out.toOwnedSlice(allocator);
    }

    pub fn clearMessages(
        self: *Self,
        session_id: []const u8,
    ) !void {
        const entries = try implList(@ptrCast(self), self.allocator, .conversation, session_id);
        defer root.freeEntries(self.allocator, entries);
        for (entries) |entry| {
            _ = implForget(@ptrCast(self), entry.key) catch {};
        }
    }

    pub fn clearAutoSaved(
        self: *Self,
        session_id: ?[]const u8,
    ) !void {
        const entries = try implList(@ptrCast(self), self.allocator, null, session_id);
        defer root.freeEntries(self.allocator, entries);
        for (entries) |entry| {
            if (!std.mem.startsWith(u8, entry.key, AUTOSAVE_KEY_PREFIX)) continue;
            _ = implForget(@ptrCast(self), entry.key) catch {};
        }
    }

    pub fn countSessions(self: *Self) !u64 {
        const entries = try implList(@ptrCast(self), self.allocator, .conversation, null);
        defer root.freeEntries(self.allocator, entries);

        var seen: std.StringHashMapUnmanaged(void) = .empty;
        defer seen.deinit(self.allocator);

        for (entries) |entry| {
            const sid = entry.session_id orelse continue;
            // Skip runtime-command-only "sessions" the way sqlite does.
            const role = roleFromMessageKey(entry.key) orelse continue;
            if (root.isRuntimeCommandRole(role)) continue;
            _ = try seen.getOrPut(self.allocator, sid);
        }
        return seen.count();
    }

    /// Extract the `<nanos>` component from either message-key scheme:
    ///   * `msg:<session>:<nanos>:<role>` — nanos between the last
    ///     two `:` separators.
    ///   * `autosave_user_<nanos>` / `autosave_assistant_<nanos>` —
    ///     digits after the role-prefix.
    /// Returns 0 on any parse failure so sort order stays deterministic.
    fn nanosFromMessageKey(key: []const u8) i128 {
        if (std.mem.startsWith(u8, key, AUTOSAVE_USER_PREFIX)) {
            return std.fmt.parseInt(i128, key[AUTOSAVE_USER_PREFIX.len..], 10) catch 0;
        }
        if (std.mem.startsWith(u8, key, AUTOSAVE_ASSISTANT_PREFIX)) {
            return std.fmt.parseInt(i128, key[AUTOSAVE_ASSISTANT_PREFIX.len..], 10) catch 0;
        }
        if (!std.mem.startsWith(u8, key, MESSAGE_KEY_PREFIX)) return 0;
        // Find the last ':' (role separator) and the one before it
        // (nanos separator) by scanning from the right.
        var last_colon: ?usize = null;
        var prev_colon: ?usize = null;
        var i: usize = key.len;
        while (i > 0) : (i -= 1) {
            if (key[i - 1] == ':') {
                if (last_colon == null) {
                    last_colon = i - 1;
                } else {
                    prev_colon = i - 1;
                    break;
                }
            }
        }
        const lc = last_colon orelse return 0;
        const pc = prev_colon orelse return 0;
        if (pc + 1 >= lc) return 0;
        return std.fmt.parseInt(i128, key[pc + 1 .. lc], 10) catch 0;
    }

    pub fn listSessions(
        self: *Self,
        allocator: Allocator,
        limit: usize,
        offset: usize,
    ) ![]root.SessionInfo {
        const entries = try implList(@ptrCast(self), self.allocator, .conversation, null);
        defer root.freeEntries(self.allocator, entries);

        // Aggregate per-session: message count plus the first and last
        // nanosecond timestamps pulled directly from the message key.
        // Key-derived nanos beat the Entry's millisecond-resolution
        // `created_at` property because a burst of messages written in
        // the same millisecond would otherwise collapse into ambiguous
        // sort order — reproducible in CI, flaky locally.
        const SessionAgg = struct {
            count: u64,
            first_ns: i128,
            last_ns: i128,
        };
        var agg: std.StringHashMapUnmanaged(SessionAgg) = .empty;
        defer {
            var it = agg.iterator();
            while (it.next()) |e| self.allocator.free(e.key_ptr.*);
            agg.deinit(self.allocator);
        }

        for (entries) |entry| {
            const sid = entry.session_id orelse continue;
            const role = roleFromMessageKey(entry.key) orelse continue;
            if (root.isRuntimeCommandRole(role)) continue;
            const ns = nanosFromMessageKey(entry.key);

            const gop = try agg.getOrPut(self.allocator, sid);
            if (!gop.found_existing) {
                gop.key_ptr.* = try self.allocator.dupe(u8, sid);
                gop.value_ptr.* = .{ .count = 1, .first_ns = ns, .last_ns = ns };
            } else {
                gop.value_ptr.count += 1;
                if (ns < gop.value_ptr.first_ns) gop.value_ptr.first_ns = ns;
                if (ns > gop.value_ptr.last_ns) gop.value_ptr.last_ns = ns;
            }
        }

        // Materialize into a sortable array (most-recent first) before
        // applying limit/offset — matches the `ORDER BY MAX(created_at)
        // DESC` clause in the sqlite engine.
        const Item = struct {
            sid: []const u8,
            count: u64,
            first_ns: i128,
            last_ns: i128,
        };
        var items: std.ArrayListUnmanaged(Item) = .empty;
        defer items.deinit(self.allocator);
        var iter = agg.iterator();
        while (iter.next()) |e| {
            try items.append(self.allocator, .{
                .sid = e.key_ptr.*,
                .count = e.value_ptr.count,
                .first_ns = e.value_ptr.first_ns,
                .last_ns = e.value_ptr.last_ns,
            });
        }
        std.mem.sort(Item, items.items, {}, struct {
            fn lt(_: void, a: Item, b: Item) bool {
                return a.last_ns > b.last_ns;
            }
        }.lt);

        var out: std.ArrayListUnmanaged(root.SessionInfo) = .empty;
        errdefer {
            for (out.items) |info| info.deinit(allocator);
            out.deinit(allocator);
        }

        const total = items.items.len;
        if (offset >= total) return try out.toOwnedSlice(allocator);
        const end = @min(offset + limit, total);

        for (items.items[offset..end]) |item| {
            const first_buf = try std.fmt.allocPrint(allocator, "{d}", .{item.first_ns});
            errdefer allocator.free(first_buf);
            const last_buf = try std.fmt.allocPrint(allocator, "{d}", .{item.last_ns});
            errdefer allocator.free(last_buf);
            const sid_copy = try allocator.dupe(u8, item.sid);
            try out.append(allocator, .{
                .session_id = sid_copy,
                .message_count = item.count,
                .first_message_at = first_buf,
                .last_message_at = last_buf,
            });
        }
        return try out.toOwnedSlice(allocator);
    }

    pub fn countDetailedMessages(
        self: *Self,
        session_id: []const u8,
    ) !u64 {
        const entries = try implList(@ptrCast(self), self.allocator, .conversation, session_id);
        defer root.freeEntries(self.allocator, entries);

        var n: u64 = 0;
        for (entries) |entry| {
            const role = roleFromMessageKey(entry.key) orelse continue;
            if (root.isRuntimeCommandRole(role)) continue;
            n += 1;
        }
        return n;
    }

    /// BM25 + graph recall over conversation messages.
    ///
    /// Every message saved via `saveMessage` lands on disk as an
    /// `Entry` node in the `.conversation` category and is indexed by
    /// lattice's BM25 FTS at the same time (inherited from `implStore`
    /// → `lattice_fts_index`). That means the moment a message is
    /// persisted, it becomes retrievable by semantic keyword match
    /// through this helper — **without** a separate prompt-time scan,
    /// without a sidecar vector store, and without any extra writes.
    ///
    /// Callers pass a free-text `query` (the user's current turn, a
    /// topic keyword, or a compacted summary); latticedb returns up to
    /// `limit` ranked matches, optionally filtered to a single
    /// `session_id`. Pass `session_id == null` to pull related turns
    /// across *every* past conversation — the graph + FTS combination
    /// is what makes this cheap compared to scanning a flat table.
    ///
    /// This is the entry point an `Agent` uses to inject relevant
    /// prior turns into the prompt on top of chronological
    /// `loadMessages` results. The SessionStore vtable intentionally
    /// does not surface it — this kind of ranked retrieval is
    /// backend-specific and not required by every engine — so
    /// adapters that want it downcast to `LatticeMemory` first.
    pub fn recallMessages(
        self: *Self,
        allocator: Allocator,
        query: []const u8,
        limit: usize,
        session_id: ?[]const u8,
    ) ![]root.MessageEntry {
        const entries = try implRecall(@ptrCast(self), allocator, query, limit, session_id);
        defer root.freeEntries(allocator, entries);

        var out: std.ArrayListUnmanaged(root.MessageEntry) = .empty;
        errdefer {
            for (out.items) |m| {
                allocator.free(m.role);
                allocator.free(m.content);
            }
            out.deinit(allocator);
        }

        for (entries) |entry| {
            // Skip hits that aren't chat messages (e.g. bootstrap templates
            // that happen to share the current FTS vocabulary).
            const role_slice = roleFromMessageKey(entry.key) orelse continue;
            const role_copy = try allocator.dupe(u8, role_slice);
            errdefer allocator.free(role_copy);
            const content_copy = try allocator.dupe(u8, entry.content);
            try out.append(allocator, .{ .role = role_copy, .content = content_copy });
        }

        return try out.toOwnedSlice(allocator);
    }

    pub fn loadMessagesDetailed(
        self: *Self,
        allocator: Allocator,
        session_id: []const u8,
        limit: usize,
        offset: usize,
    ) ![]root.DetailedMessageEntry {
        const entries = try implList(@ptrCast(self), self.allocator, .conversation, session_id);
        defer root.freeEntries(self.allocator, entries);

        std.mem.sort(root.MemoryEntry, entries, {}, struct {
            fn lt(_: void, a: root.MemoryEntry, b: root.MemoryEntry) bool {
                return std.mem.lessThan(u8, a.key, b.key);
            }
        }.lt);

        var out: std.ArrayListUnmanaged(root.DetailedMessageEntry) = .empty;
        errdefer {
            for (out.items) |m| {
                allocator.free(m.role);
                allocator.free(m.content);
                allocator.free(m.created_at);
            }
            out.deinit(allocator);
        }

        var index: usize = 0;
        var emitted: usize = 0;
        for (entries) |entry| {
            const role_slice = roleFromMessageKey(entry.key) orelse continue;
            if (root.isRuntimeCommandRole(role_slice)) continue;
            if (index < offset) {
                index += 1;
                continue;
            }
            if (emitted >= limit) break;
            index += 1;
            emitted += 1;
            const role_copy = try allocator.dupe(u8, role_slice);
            errdefer allocator.free(role_copy);
            const content_copy = try allocator.dupe(u8, entry.content);
            errdefer allocator.free(content_copy);
            const ts_copy = try allocator.dupe(u8, entry.timestamp);
            try out.append(allocator, .{
                .role = role_copy,
                .content = content_copy,
                .created_at = ts_copy,
            });
        }
        return try out.toOwnedSlice(allocator);
    }

    // ── SessionStore vtable glue ────────────────────────────────────

    fn implSessionSaveMessage(ptr: *anyopaque, session_id: []const u8, role: []const u8, content: []const u8) anyerror!void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.saveMessage(session_id, role, content);
    }

    fn implSessionLoadMessages(ptr: *anyopaque, allocator: Allocator, session_id: []const u8) anyerror![]root.MessageEntry {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.loadMessages(allocator, session_id);
    }

    fn implSessionClearMessages(ptr: *anyopaque, session_id: []const u8) anyerror!void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.clearMessages(session_id);
    }

    fn implSessionClearAutoSaved(ptr: *anyopaque, session_id: ?[]const u8) anyerror!void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.clearAutoSaved(session_id);
    }

    fn implSessionCountSessions(ptr: *anyopaque) anyerror!u64 {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.countSessions();
    }

    fn implSessionListSessions(ptr: *anyopaque, allocator: Allocator, limit: usize, offset: usize) anyerror![]root.SessionInfo {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.listSessions(allocator, limit, offset);
    }

    fn implSessionCountDetailedMessages(ptr: *anyopaque, session_id: []const u8) anyerror!u64 {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.countDetailedMessages(session_id);
    }

    fn implSessionLoadMessagesDetailed(ptr: *anyopaque, allocator: Allocator, session_id: []const u8, limit: usize, offset: usize) anyerror![]root.DetailedMessageEntry {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.loadMessagesDetailed(allocator, session_id, limit, offset);
    }

    const session_vtable = root.SessionStore.VTable{
        .saveMessage = &implSessionSaveMessage,
        .loadMessages = &implSessionLoadMessages,
        .clearMessages = &implSessionClearMessages,
        .clearAutoSaved = &implSessionClearAutoSaved,
        .countSessions = &implSessionCountSessions,
        .listSessions = &implSessionListSessions,
        .countDetailedMessages = &implSessionCountDetailedMessages,
        .loadMessagesDetailed = &implSessionLoadMessagesDetailed,
        // `saveUsage` / `loadUsage` are opt-in on the vtable — leave
        // them unset and the generic wrapper will surface NotSupported,
        // matching engines like clickhouse that don't track usage.
    };

    pub fn sessionStore(self: *Self) root.SessionStore {
        return .{ .ptr = @ptrCast(self), .vtable = &session_vtable };
    }
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

test "latticedb session store round-trips messages across reopens" {
    // Regression: before this, latticedb had `supports_session_store=false`
    // so `Agent.session_store` was null for latticedb installs and every
    // `nullclaw agent` run started with a blank conversation context —
    // even though bootstrap entries were present on disk. The new
    // `sessionStore()` surfaces saveMessage/loadMessages/clearMessages
    // on top of Entry nodes keyed by `msg:<session>:<nanos>:<role>`.
    // This test proves:
    //   1. messages survive process-level reopen (via the same label
    //      scan that rebuilds `key_to_node`);
    //   2. loadMessages returns them in insertion order;
    //   3. multi-KiB assistant turns go through the chunk-node path so
    //      large bodies reassemble byte-for-byte;
    //   4. countSessions / listSessions / countDetailedMessages /
    //      loadMessagesDetailed / clearMessages all behave consistently;
    //   5. clearAutoSaved only targets keys with the `autosave_` prefix.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const paths = try openTmp(&tmp);
    defer std.testing.allocator.free(paths.path);
    defer std.testing.allocator.free(paths.base);

    // Build a ~6 KiB assistant reply so we know the chunk pipeline
    // is actually exercised by saveMessage.
    const big_reply = try std.testing.allocator.alloc(u8, 6000);
    defer std.testing.allocator.free(big_reply);
    for (big_reply, 0..) |*b, i| b.* = 'a' + @as(u8, @intCast(i % 26));

    {
        var mem = try LatticeMemory.init(std.testing.allocator, paths.path);
        defer mem.deinit();
        const store = mem.sessionStore();

        try store.saveMessage("s1", "user", "hi");
        try store.saveMessage("s1", "assistant", big_reply);
        try store.saveMessage("s1", "user", "follow-up");
        try store.saveMessage("s2", "user", "hello from a second session");

        // An autosave-style Memory entry that clearAutoSaved should later
        // remove but `clearMessages("s1")` must leave alone.
        try mem.memory().store("autosave_user_s1_123", "scratch", .core, "s1");
    }

    // Reopen: rebuilt caches should expose the same messages.
    var mem = try LatticeMemory.init(std.testing.allocator, paths.path);
    defer mem.deinit();
    const store = mem.sessionStore();

    {
        const msgs = try store.loadMessages(std.testing.allocator, "s1");
        defer root.freeMessages(std.testing.allocator, msgs);
        try std.testing.expectEqual(@as(usize, 3), msgs.len);
        try std.testing.expectEqualStrings("user", msgs[0].role);
        try std.testing.expectEqualStrings("hi", msgs[0].content);
        try std.testing.expectEqualStrings("assistant", msgs[1].role);
        try std.testing.expectEqualSlices(u8, big_reply, msgs[1].content);
        try std.testing.expectEqualStrings("user", msgs[2].role);
        try std.testing.expectEqualStrings("follow-up", msgs[2].content);
    }

    try std.testing.expectEqual(@as(u64, 2), try store.countSessions());
    try std.testing.expectEqual(@as(u64, 3), try store.countDetailedMessages("s1"));

    {
        const sessions = try store.listSessions(std.testing.allocator, 10, 0);
        defer root.freeSessionInfos(std.testing.allocator, sessions);
        try std.testing.expectEqual(@as(usize, 2), sessions.len);
        // Most recent session first: s2 was written after the s1 burst.
        try std.testing.expectEqualStrings("s2", sessions[0].session_id);
        try std.testing.expectEqual(@as(u64, 1), sessions[0].message_count);
        try std.testing.expectEqualStrings("s1", sessions[1].session_id);
        try std.testing.expectEqual(@as(u64, 3), sessions[1].message_count);
    }

    {
        const detailed = try store.loadMessagesDetailed(std.testing.allocator, "s1", 10, 0);
        defer root.freeDetailedMessages(std.testing.allocator, detailed);
        try std.testing.expectEqual(@as(usize, 3), detailed.len);
        try std.testing.expect(detailed[0].created_at.len > 0);
    }

    // clearAutoSaved nukes the scratch entry but leaves chat messages.
    try store.clearAutoSaved(null);
    {
        const got = try mem.memory().get(std.testing.allocator, "autosave_user_s1_123");
        try std.testing.expect(got == null);
        try std.testing.expectEqual(@as(u64, 3), try store.countDetailedMessages("s1"));
    }

    // clearMessages drops the whole s1 conversation.
    try store.clearMessages("s1");
    {
        const msgs = try store.loadMessages(std.testing.allocator, "s1");
        defer root.freeMessages(std.testing.allocator, msgs);
        try std.testing.expectEqual(@as(usize, 0), msgs.len);
    }
    try std.testing.expectEqual(@as(u64, 1), try store.countSessions());
}

test "latticedb session messages feed BM25 recall and cross-session search" {
    // The whole point of backing chat history with latticedb instead
    // of a flat sqlite table is that every saved message lands in the
    // BM25 FTS index for free (via implStore → lattice_fts_index), so
    // the agent can recall a single topic across every past session
    // without running a separate vector/search pipeline. This test
    // exercises that end-to-end:
    //
    //   1. Multi-session message writes go through saveMessage.
    //   2. `recallMessages("zebra", session_id)` surfaces the matching
    //      turn scoped to a single session.
    //   3. `recallMessages("zebra", null)` walks the graph across
    //      sessions and returns the same hit.
    //   4. Messages in the conversation category are also visible to
    //      the generic `memory.recall()` entry point so existing
    //      retrieval pipelines (which use `mem.recall`) keep working.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const paths = try openTmp(&tmp);
    defer std.testing.allocator.free(paths.path);
    defer std.testing.allocator.free(paths.base);

    var mem = try LatticeMemory.init(std.testing.allocator, paths.path);
    defer mem.deinit();
    const store = mem.sessionStore();

    // Session A: normal chat + one message containing a rare token.
    try store.saveMessage("sessA", "user", "tell me about my travels");
    try store.saveMessage("sessA", "assistant", "last winter you went to see zebra herds in tanzania");
    try store.saveMessage("sessA", "user", "great, what else");

    // Session B: noise with no overlap.
    try store.saveMessage("sessB", "user", "help me draft a quarterly SMM plan");
    try store.saveMessage("sessB", "assistant", "start with audience segmentation");

    // In-session BM25 lookup — should land on the zebra turn.
    {
        const hits = try mem.recallMessages(std.testing.allocator, "zebra", 5, "sessA");
        defer root.freeMessages(std.testing.allocator, hits);
        try std.testing.expect(hits.len >= 1);
        var found = false;
        for (hits) |h| {
            if (std.mem.indexOf(u8, h.content, "zebra") != null) {
                found = true;
                try std.testing.expectEqualStrings("assistant", h.role);
                break;
            }
        }
        try std.testing.expect(found);
    }

    // Cross-session BM25 lookup (session_id = null) still returns the
    // zebra turn even though it lives inside sessA. That is the graph
    // + FTS combination — one call, no SQL table scans.
    {
        const hits = try mem.recallMessages(std.testing.allocator, "zebra", 5, null);
        defer root.freeMessages(std.testing.allocator, hits);
        try std.testing.expect(hits.len >= 1);
        try std.testing.expect(std.mem.indexOf(u8, hits[0].content, "zebra") != null);
    }

    // The generic Memory.recall entry point sees the same messages
    // because they're regular Entry nodes with FTS coverage — the
    // retrieval pipeline in `mem_rt` calls into it for free.
    {
        const hits = try mem.memory().recall(std.testing.allocator, "SMM", 5, null);
        defer root.freeEntries(std.testing.allocator, hits);
        try std.testing.expect(hits.len >= 1);
        try std.testing.expect(std.mem.indexOf(u8, hits[0].content, "SMM") != null);
    }
}

test "latticedb memory rebuilds key/category/session caches on reopen" {
    // Regression: before the `lattice_get_nodes_by_label`-backed
    // rebuild, a process that opened an existing LatticeDB would
    // create duplicate Entry nodes for keys that were already on
    // disk. Today the rebuild is lazy — `init()` itself does not
    // scan, but the first write (`store`) / point read (`get`) /
    // delete (`forget`) triggers `ensureCachesLoaded`, so the
    // observable behavior (count, get, overwrite) is still correct.
    // This test exercises that behavior through the public Memory
    // API rather than reaching into private maps.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const paths = try openTmp(&tmp);
    defer std.testing.allocator.free(paths.path);
    defer std.testing.allocator.free(paths.base);

    var initial_entry_id: NodeId = 0;
    {
        var mem = try LatticeMemory.init(std.testing.allocator, paths.path);
        defer mem.deinit();
        const m = mem.memory();

        try m.store("alpha", "first entry content", .core, "sess-1");
        try m.store("beta", "second entry content", .core, "sess-1");
        try m.store("gamma", "third entry content", .daily, "sess-2");

        try std.testing.expectEqual(@as(usize, 3), try m.count());
        initial_entry_id = mem.key_to_node.get("alpha").?;
    }

    // Reopen — init() leaves the caches empty; the first write below
    // triggers `ensureCachesLoaded` which scans the on-disk Entry /
    // Category / Session labels via `lattice_get_nodes_by_label` and
    // repopulates every in-memory map before any overwrite check runs.
    var mem = try LatticeMemory.init(std.testing.allocator, paths.path);
    defer mem.deinit();
    const m = mem.memory();

    // Fast-path `count()` hits disk via `lattice_get_nodes_by_label`
    // without loading the full cache — safe to call before any
    // mutation and should already report the on-disk total.
    try std.testing.expectEqual(@as(usize, 3), try m.count());

    // Touch a key through the public API so the caches get populated,
    // then assert that the internal map was refilled from disk with
    // the original node ids (not reassigned).
    {
        const touch = try m.get(std.testing.allocator, "alpha");
        try std.testing.expect(touch != null);
        touch.?.deinit(std.testing.allocator);
    }
    const reloaded_entry_id = mem.key_to_node.get("alpha").?;
    try std.testing.expectEqual(initial_entry_id, reloaded_entry_id);

    {
        const got = try m.get(std.testing.allocator, "alpha");
        try std.testing.expect(got != null);
        defer got.?.deinit(std.testing.allocator);
        try std.testing.expectEqualStrings("first entry content", got.?.content);
        try std.testing.expect(got.?.category.eql(.core));
        try std.testing.expect(got.?.session_id != null);
        try std.testing.expectEqualStrings("sess-1", got.?.session_id.?);
    }

    // Categories/sessions seen on disk must be reused, not re-created.
    try std.testing.expect(mem.category_to_node.contains("core"));
    try std.testing.expect(mem.category_to_node.contains("daily"));
    try std.testing.expect(mem.session_to_node.contains("sess-1"));
    try std.testing.expect(mem.session_to_node.contains("sess-2"));

    // Overwrite path: `store` on an existing key must find the cached
    // node, delete it (cascading its chunks), and re-insert. Without
    // the rebuild, the cache lookup would miss and we'd end up with
    // two Entry nodes with the same `key` string.
    try m.store("alpha", "overwritten content", .core, "sess-1");
    try std.testing.expectEqual(@as(usize, 3), try m.count());

    {
        const got = try m.get(std.testing.allocator, "alpha");
        try std.testing.expect(got != null);
        defer got.?.deinit(std.testing.allocator);
        try std.testing.expectEqualStrings("overwritten content", got.?.content);
    }
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
