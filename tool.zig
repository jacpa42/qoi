const std = @import("std");
const qoi = @import("qoi.zig");

pub fn main() !void {
    var debug_alloc = std.heap.DebugAllocator(.{}).init;
    defer _ = debug_alloc.deinit();
    const alloc = debug_alloc.allocator();

    const path: [:0]const u8 = blk: {
        var args = std.process.args();
        _ = args.next();
        break :blk args.next() orelse return error.ExpectedQoiFile;
    };

    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    var buf: [1024]u8 = undefined;
    var freader = file.reader(&buf);

    var raw_img = try qoi.decodeReader(alloc, &freader.interface);
    defer raw_img.deinit(alloc);

    std.log.info("width: {}", .{raw_img.width});
    std.log.info("height: {}", .{raw_img.height});
    std.log.info("channels: {t}", .{raw_img.channels});
    std.log.info("colorspace: {t}", .{raw_img.colorspace});
    std.log.info("pixel count: {}", .{raw_img.pixels.len});

    var stdout = std.fs.File.stdout();
    var writer = stdout.writer(&buf);

    try qoi.encodeWriter(raw_img, &writer.interface);
}
