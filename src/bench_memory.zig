//! Memory backend microbenchmark.
//!
//! Compiled by `zig build bench` and runs a fixed scenario matrix against
//! every memory engine that the current build links in (always sqlite,
//! plus latticedb when `-Dengines=...,latticedb` is enabled). The goal is
//! NOT to crown a winner — the workloads are deliberately tiny so they
//! fit on a laptop in seconds — but to give a rough order-of-magnitude
//! comparison of write/get/recall/list latency and on-disk footprint as
//! the row count and per-row content size grow.
//!
//! Usage:
//!   zig build bench                                # default matrix
//!   zig build bench -- --entries 5000              # override row count
//!   zig build bench -- --content-size 4096         # override row size
//!   zig build bench -- --backend sqlite|latticedb  # restrict to one
//!
//! Output is a plain ASCII table on stdout. Each backend gets its own
//! private workspace under `std.testing.tmpDir(.{})` so the runs do not
//! step on each other.

const std = @import("std");
const builtin = @import("builtin");
const yc = @import("nullclaw");
const build_options = @import("build_options");

const Memory = yc.memory.Memory;
const MemoryEntry = yc.memory.MemoryEntry;

const Args = struct {
    entries: usize = 1000,
    content_size: usize = 512,
    backend: enum { both, sqlite, latticedb } = .both,
    recall_queries: usize = 50,
    seed: u64 = 0xc0ffee,
};

fn parseArgs(allocator: std.mem.Allocator) !Args {
    var args = Args{};
    var it = try std.process.argsWithAllocator(allocator);
    defer it.deinit();
    _ = it.next(); // exe
    while (it.next()) |raw| {
        if (std.mem.eql(u8, raw, "--entries")) {
            const v = it.next() orelse return error.MissingValue;
            args.entries = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, raw, "--content-size")) {
            const v = it.next() orelse return error.MissingValue;
            args.content_size = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, raw, "--backend")) {
            const v = it.next() orelse return error.MissingValue;
            if (std.mem.eql(u8, v, "sqlite")) {
                args.backend = .sqlite;
            } else if (std.mem.eql(u8, v, "latticedb")) {
                args.backend = .latticedb;
            } else if (std.mem.eql(u8, v, "both")) {
                args.backend = .both;
            } else {
                return error.UnknownBackend;
            }
        } else if (std.mem.eql(u8, raw, "--recall-queries")) {
            const v = it.next() orelse return error.MissingValue;
            args.recall_queries = try std.fmt.parseInt(usize, v, 10);
        }
    }
    return args;
}

const Result = struct {
    backend: []const u8,
    entries: usize,
    content_size: usize,
    open_us: u64 = 0,
    store_total_ms: u64 = 0,
    store_per_op_us: u64 = 0,
    get_total_ms: u64 = 0,
    get_per_op_us: u64 = 0,
    recall_total_ms: u64 = 0,
    recall_per_op_us: u64 = 0,
    list_total_ms: u64 = 0,
    reopen_us: u64 = 0,
    disk_bytes: u64 = 0,
};

/// Build a content blob of the requested size whose tokens are
/// reasonably distinct so BM25 actually has something to index. We mix
/// a per-row "topic" word with a stable shared vocabulary so recall
/// queries hit predictable rows.
fn fillContent(buf: []u8, row: usize) void {
    const topic_words = [_][]const u8{
        "alpha",  "bravo",  "charlie", "delta",  "echo",   "foxtrot", "golf",
        "hotel",  "india",  "juliet",  "kilo",   "lima",   "mike",    "november",
        "oscar",  "papa",   "quebec",  "romeo",  "sierra", "tango",   "uniform",
        "victor", "whisky", "xray",    "yankee", "zulu",
    };
    const filler_words = [_][]const u8{
        "the",    "quick", "brown", "fox",   "jumps",  "over",   "lazy",
        "dog",    "loop",  "table", "graph", "memory", "search", "query",
        "vector", "index", "hash",  "cache", "page",   "node",   "edge",
    };
    var pos: usize = 0;
    var i: usize = 0;
    while (pos + 16 <= buf.len) : (i += 1) {
        const word = if (i == 0)
            topic_words[row % topic_words.len]
        else
            filler_words[(row + i) % filler_words.len];
        const take = @min(word.len, buf.len - pos);
        @memcpy(buf[pos..][0..take], word[0..take]);
        pos += take;
        if (pos < buf.len) {
            buf[pos] = ' ';
            pos += 1;
        }
    }
    while (pos < buf.len) : (pos += 1) buf[pos] = 'x';
}

fn runOne(
    allocator: std.mem.Allocator,
    name: []const u8,
    open_fn: *const fn (std.mem.Allocator, []const u8) anyerror!Memory,
    deinit_fn: *const fn (Memory) void,
    args: Args,
    workspace: []const u8,
    db_path: []const u8,
) !Result {
    var result = Result{
        .backend = name,
        .entries = args.entries,
        .content_size = args.content_size,
    };
    _ = workspace;

    // ── open ─────────────────────────────────────────────────────
    var t = try std.time.Timer.start();
    var mem = try open_fn(allocator, db_path);
    result.open_us = t.read() / std.time.ns_per_us;

    // Build a single content buffer we can mutate per row to avoid
    // re-allocating; copy bytes in place.
    const content = try allocator.alloc(u8, args.content_size);
    defer allocator.free(content);

    // ── store ────────────────────────────────────────────────────
    t.reset();
    var i: usize = 0;
    while (i < args.entries) : (i += 1) {
        fillContent(content, i);
        var key_buf: [64]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "row_{d}", .{i});
        try mem.store(key, content, .core, null);
    }
    const store_ns = t.read();
    result.store_total_ms = store_ns / std.time.ns_per_ms;
    result.store_per_op_us = (store_ns / std.time.ns_per_us) / @max(args.entries, 1);

    // ── get (random access) ──────────────────────────────────────
    var rng = std.Random.DefaultPrng.init(args.seed);
    const r = rng.random();
    t.reset();
    var hit: usize = 0;
    var j: usize = 0;
    while (j < args.entries) : (j += 1) {
        const idx = r.intRangeAtMost(usize, 0, args.entries - 1);
        var key_buf: [64]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "row_{d}", .{idx});
        if (try mem.get(allocator, key)) |entry| {
            hit += 1;
            entry.deinit(allocator);
        }
    }
    const get_ns = t.read();
    result.get_total_ms = get_ns / std.time.ns_per_ms;
    result.get_per_op_us = (get_ns / std.time.ns_per_us) / @max(args.entries, 1);

    // ── recall (BM25 keyword search) ─────────────────────────────
    const probe_terms = [_][]const u8{
        "alpha", "bravo",  "charlie", "delta", "echo",   "foxtrot",
        "tango", "victor", "whisky",  "xray",  "yankee", "zulu",
    };
    t.reset();
    var k: usize = 0;
    while (k < args.recall_queries) : (k += 1) {
        const term = probe_terms[k % probe_terms.len];
        const out = mem.recall(allocator, term, 5, null) catch continue;
        defer yc.memory.freeEntries(allocator, out);
    }
    const recall_ns = t.read();
    result.recall_total_ms = recall_ns / std.time.ns_per_ms;
    result.recall_per_op_us = (recall_ns / std.time.ns_per_us) / @max(args.recall_queries, 1);

    // ── list (full category scan) ────────────────────────────────
    t.reset();
    const listed = try mem.list(allocator, .core, null);
    const list_ns = t.read();
    yc.memory.freeEntries(allocator, listed);
    result.list_total_ms = list_ns / std.time.ns_per_ms;

    // ── disk size before close ───────────────────────────────────
    if (std.fs.openFileAbsolute(db_path, .{})) |f| {
        defer f.close();
        const stat = f.stat() catch null;
        if (stat) |s| result.disk_bytes = s.size;
    } else |_| {}

    deinit_fn(mem);

    // ── reopen latency ───────────────────────────────────────────
    t.reset();
    const mem2 = try open_fn(allocator, db_path);
    result.reopen_us = t.read() / std.time.ns_per_us;
    deinit_fn(mem2);

    return result;
}

fn openSqlite(allocator: std.mem.Allocator, db_path: []const u8) !Memory {
    const path_z = try allocator.dupeZ(u8, db_path);
    defer allocator.free(path_z);
    const impl = try allocator.create(yc.memory.SqliteMemory);
    impl.* = try yc.memory.SqliteMemory.init(allocator, path_z.ptr);
    return impl.memory();
}

fn deinitSqlite(mem: Memory) void {
    const impl: *yc.memory.SqliteMemory = @ptrCast(@alignCast(mem.ptr));
    const allocator = impl.allocator;
    impl.deinit();
    allocator.destroy(impl);
}

fn openLattice(allocator: std.mem.Allocator, db_path: []const u8) !Memory {
    if (!build_options.enable_memory_latticedb) return error.LatticeDbNotEnabled;
    const impl = try allocator.create(yc.memory.LatticeMemory);
    impl.* = try yc.memory.LatticeMemory.init(allocator, db_path);
    return impl.memory();
}

fn deinitLattice(mem: Memory) void {
    const impl: *yc.memory.LatticeMemory = @ptrCast(@alignCast(mem.ptr));
    const allocator = impl.allocator;
    impl.deinit();
    allocator.destroy(impl);
}

fn humanBytes(bytes: u64) struct { value: f64, unit: []const u8 } {
    const mb: f64 = @floatFromInt(bytes);
    if (bytes >= 1024 * 1024) return .{ .value = mb / (1024.0 * 1024.0), .unit = "MiB" };
    if (bytes >= 1024) return .{ .value = mb / 1024.0, .unit = "KiB" };
    return .{ .value = mb, .unit = "B" };
}

fn printResult(stdout: anytype, r: Result) !void {
    const disk = humanBytes(r.disk_bytes);
    try stdout.print(
        "  {s: <10} | {d: >7} | {d: >5} B | open {d: >6} µs | store {d: >6} ms ({d: >5} µs/op) | get {d: >6} ms ({d: >5} µs/op) | recall {d: >5} ms ({d: >5} µs/q) | list {d: >5} ms | reopen {d: >6} µs | disk {d: >6.1} {s}\n",
        .{
            r.backend,
            r.entries,
            r.content_size,
            r.open_us,
            r.store_total_ms,
            r.store_per_op_us,
            r.get_total_ms,
            r.get_per_op_us,
            r.recall_total_ms,
            r.recall_per_op_us,
            r.list_total_ms,
            r.reopen_us,
            disk.value,
            disk.unit,
        },
    );
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = parseArgs(allocator) catch |err| {
        std.debug.print("argument error: {s}\n", .{@errorName(err)});
        std.process.exit(2);
    };

    var stdout_buf: [4096]u8 = undefined;
    var bw = std.fs.File.stdout().writer(&stdout_buf);
    const stdout = &bw.interface;

    try stdout.print(
        "memory bench: entries={d} content_size={d} recall_queries={d}\n",
        .{ args.entries, args.content_size, args.recall_queries },
    );
    try stdout.flush();

    // Build a temp workspace for each backend so they don't collide.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_root = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_root);

    var rows: std.ArrayListUnmanaged(Result) = .empty;
    defer rows.deinit(allocator);

    if (args.backend == .both or args.backend == .sqlite) {
        const sqlite_path = try std.fs.path.join(allocator, &.{ tmp_root, "bench_sqlite.db" });
        defer allocator.free(sqlite_path);
        const result = runOne(
            allocator,
            "sqlite",
            openSqlite,
            deinitSqlite,
            args,
            tmp_root,
            sqlite_path,
        ) catch |err| {
            try stdout.print("  sqlite skipped: {s}\n", .{@errorName(err)});
            try stdout.flush();
            return;
        };
        try rows.append(allocator, result);
    }

    if (args.backend == .both or args.backend == .latticedb) {
        if (build_options.enable_memory_latticedb) {
            const lattice_path = try std.fs.path.join(allocator, &.{ tmp_root, "bench_lattice.db" });
            defer allocator.free(lattice_path);
            const result = runOne(
                allocator,
                "latticedb",
                openLattice,
                deinitLattice,
                args,
                tmp_root,
                lattice_path,
            ) catch |err| {
                try stdout.print("  latticedb skipped: {s}\n", .{@errorName(err)});
                try stdout.flush();
                return;
            };
            try rows.append(allocator, result);
        } else {
            try stdout.print("  latticedb skipped: build without -Dengines=...,latticedb\n", .{});
        }
    }

    try stdout.print("\n", .{});
    for (rows.items) |r| try printResult(stdout, r);
    try stdout.flush();
}
