//! LatticeDB memory backend.
//!
//! Stores agent memory as a property graph:
//!   - Entry nodes   (label "Entry")    — properties: key, content, created_at
//!   - Category nodes (label "Category") — property: name
//!   - Session nodes  (label "Session")  — property: id
//!   - BELONGS_TO_CATEGORY edges from each Entry to its Category
//!   - IN_SESSION edges from each Entry to its Session (when session_id is set)
//!   - FTS index on Entry.content for recall() via BM25
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
const CATEGORY_LABEL: [:0]const u8 = "Category";
const SESSION_LABEL: [:0]const u8 = "Session";
const BELONGS_TO_CATEGORY: [:0]const u8 = "BELONGS_TO_CATEGORY";
const IN_SESSION: [:0]const u8 = "IN_SESSION";
const PROP_KEY: [:0]const u8 = "key";
const PROP_CONTENT: [:0]const u8 = "content";
const PROP_CREATED_AT: [:0]const u8 = "created_at";
const PROP_NAME: [:0]const u8 = "name";
const PROP_ID: [:0]const u8 = "id";

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
            .page_size = 4096,
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

        const content = (try readStringProperty(allocator, txn, entry_node_id, PROP_CONTENT)) orelse return null;
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
        try setNodePropertyString(txn, entry_id, PROP_CONTENT, content);
        try setNodePropertyInt(txn, entry_id, PROP_CREATED_AT, std.time.milliTimestamp());

        const cat_node = try self.getOrCreateCategoryNode(txn, category.toString());
        var cat_edge: EdgeId = 0;
        try mapCErr(c.lattice_edge_create(txn, entry_id, cat_node, BELONGS_TO_CATEGORY.ptr, &cat_edge));

        if (session_id) |sid| {
            const sess_node = try self.getOrCreateSessionNode(txn, sid);
            var sess_edge: EdgeId = 0;
            try mapCErr(c.lattice_edge_create(txn, entry_id, sess_node, IN_SESSION.ptr, &sess_edge));
        }

        if (content.len > 0) {
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
