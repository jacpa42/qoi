const std = @import("std");
const qoi = @import("qoi.zig");

pub fn main() !void {
    var debug_alloc = std.heap.DebugAllocator(.{}).init;
    defer _ = debug_alloc.deinit();
    const alloc = debug_alloc.allocator();

    var flipx: bool = false;
    var flipy: bool = false;
    var pbuf: [2][:0]const u8 = undefined;
    var paths = std.ArrayList([:0]const u8).initBuffer(&pbuf);

    {
        var args = std.process.args();
        _ = args.next();
        while (args.next()) |arg| {
            const cmp = std.mem.eql;
            if (cmp(u8, arg, "--flipx") or
                cmp(u8, arg, "-flipx"))
            {
                flipx = true;
                continue;
            } else if (cmp(u8, arg, "--flipy") or
                cmp(u8, arg, "-flipy"))
            {
                flipy = true;
            } else {
                try paths.appendBounded(arg);
            }
        }
    }

    if (paths.items.len < 1) {
        return error.ExpectedInputFile;
    }
    if (paths.items.len == 2 and std.mem.eql(u8, paths.items[0], paths.items[1])) {
        return error.InputAndOutputFilesCannotBeTheSame;
    }

    // zig fmt: off
    const in_path = paths.items[0];
    const opts = qoi.Options{
        .flip =
            if (flipx and flipy) .xy
            else if (flipx and !flipy) .x
            else if (!flipx and flipy) .y
            else .none,
    };
    // zig fmt: on

    const file = try std.fs.cwd().openFile(in_path, .{});
    defer file.close();

    var iobuf: [8 * 1024]u8 = undefined;
    var freader = file.reader(&iobuf);

    var raw_img = try qoi.decodeReader(alloc, &freader.interface, opts);
    defer raw_img.deinit(alloc);

    std.log.info("width: {}", .{raw_img.width});
    std.log.info("height: {}", .{raw_img.height});
    std.log.info("channels: {t}", .{raw_img.channels});
    std.log.info("colour space: {t}", .{raw_img.colorspace});
    std.log.info("memory: {}Kb", .{raw_img.pixel_data.len / 1000});

    if (paths.items.len == 2) {
        const out_path = paths.items[1];
        var stdout = try std.fs.cwd().createFile(out_path, .{});
        var writer = stdout.writer(&iobuf);
        try qoi.encodeWriter(raw_img, &writer.interface);
    }
}
