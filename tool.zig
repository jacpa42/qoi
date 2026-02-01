const std = @import("std");
const qoi = @import("qoi.zig");

pub fn main() !void {
    var debug_alloc = std.heap.DebugAllocator(.{}).init;
    defer _ = debug_alloc.deinit();
    const alloc = debug_alloc.allocator();

    var flipx: bool = false;
    var flipy: bool = false;
    var path_opt: ?[:0]const u8 = null;

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
                path_opt = arg;
            }
        }
    }

    const path = path_opt orelse return error.ExpectedQoiFilePath;
    const opts = qoi.Options{
        .flip = if (flipx and flipy)
            qoi.Options.Flip.xy
        else if (flipx and !flipy)
            qoi.Options.Flip.x
        else if (!flipx and flipy)
            qoi.Options.Flip.y
        else
            qoi.Options.Flip.none,
    };

    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    var buf: [1024]u8 = undefined;
    var freader = file.reader(&buf);

    var raw_img = try qoi.decodeReader(alloc, &freader.interface, opts);
    defer raw_img.deinit(alloc);

    std.log.info("width: {}", .{raw_img.width});
    std.log.info("height: {}", .{raw_img.height});
    std.log.info("channels: {t}", .{raw_img.channels});
    std.log.info("colorspace: {t}", .{raw_img.colorspace});
    std.log.info("memory: {}Kb", .{raw_img.pixels.len / 1000});

    var stdout = std.fs.File.stdout();
    var writer = stdout.writer(&buf);

    try qoi.encodeWriter(raw_img, &writer.interface, .{});
}
