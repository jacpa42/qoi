const std = @import("std");
const Allocator = std.mem.Allocator;

/// A parsed QOI image. Does not hold onto an allocator. Use `image.deinit(alloc)` to free memory.
pub const Qoi = struct {
    width: u32,
    height: u32,
    channels: Channels,
    colorspace: ColorSpace,

    /// Pixel array list. Memory is externally managed.
    pixels: []rgba,

    pub fn deinit(self: *@This(), alloc: Allocator) void {
        alloc.free(self.pixels);
        self.* = undefined;
    }
};

pub const Channels = enum(u8) { RGB = 3, RGBA = 4 };
pub const ColorSpace = enum(u8) { SRGB = 0, LINEAR = 1 };

/// Simple rgba color.
pub const rgba = @Vector(4, u8);
inline fn hash(color: rgba) u8 {
    return 0x3f & @reduce(.Add, color *% rgba{ 3, 5, 7, 11 });
}

const QOI_OP_RGB = 0xfe;
const QOI_OP_RGBA = 0xff;
const QOI_OP_INDEX = 0x00;
const QOI_OP_DIFF = 0x40;
const QOI_OP_LUMA = 0x80;
const QOI_OP_RUN = 0xc0;

const QOI_MAGIC = "qoif";
const QOI_EOF = "\x00" ** 7 ++ "\x01";
const QOI_HEADER_SIZE = 14;

/// Parses bytes from the buffer into `QOI`.
///
/// Needs allocator to put the pixel data. Use `deinit(alloc)` do deinitialize.
pub fn decode(
    alloc: Allocator,
    raw_bytes: []const u8,
) error{
    EndOfStream,
    InvalidFileFormat,
    OutOfMemory,
    InvalidNumberOfChannels,
    InvalidColorSpaceDescription,
}!Qoi {
    if (raw_bytes.len <= QOI_HEADER_SIZE) return error.EndOfStream;
    if (!std.mem.eql(u8, raw_bytes[0..4], QOI_MAGIC)) return error.InvalidFileFormat;

    const channels: Channels = switch (raw_bytes[12]) {
        0b011 => .RGB,
        0b100 => .RGBA,
        else => return error.InvalidNumberOfChannels,
    };

    const colorspace: ColorSpace = switch (raw_bytes[13]) {
        0b0 => .SRGB,
        0b1 => .LINEAR,
        else => return error.InvalidColorSpaceDescription,
    };

    var image: Qoi = undefined;
    image.width = btn(raw_bytes[4..8].*);
    image.height = btn(raw_bytes[8..12].*);
    image.channels = channels;
    image.colorspace = colorspace;

    image.pixels = try alloc.alloc(rgba, @as(usize, image.width) * @as(usize, image.height));
    errdefer alloc.free(image.pixels);

    var color_lut: [64]rgba = @splat(@splat(0));
    var pix = rgba{ 0xff, 0x00, 0x00, 0x00 };
    var data = raw_bytes[QOI_HEADER_SIZE..];
    var run: usize = 0;

    for (image.pixels) |*pixel| {
        if (run > 0) {
            run -= 1;
        } else if (data.len > QOI_EOF.len) {
            const byte = data[0];
            data = data[1..];

            if (byte == QOI_OP_RGB) {
                pix[0] = data[0];
                pix[1] = data[1];
                pix[2] = data[2];
                data = data[3..];
            } else if (byte == QOI_OP_RGBA) {
                pix = data[0..4].*;
                data = data[4..];
            } else if (byte & 0xc0 == QOI_OP_INDEX) {
                pix = color_lut[byte];
            } else if (byte & 0xc0 == QOI_OP_DIFF) {
                var diff: rgba = @splat(byte);
                diff >>= rgba{ 4, 2, 0, 0 };
                diff &= @splat(3);
                diff -= @splat(2);
                diff[3] = 0;
                pix +%= diff;
            } else if (byte & 0xc0 == QOI_OP_LUMA) {
                const dg = (byte & 63) -% 32;
                pix +%= rgba{
                    (data[0] >> 4) -% 8 +% dg,
                    dg,
                    (data[0] & 15) -% 8 +% dg,
                    0,
                };
                data = data[1..];
            } else if (byte & 0xc0 == QOI_OP_RUN) {
                run = byte & 0x3f;
            }

            color_lut[hash(pix)] = pix;
        }

        pixel.* = pix;
    }

    return image;
}

/// Parses bytes from reader into `QOI`.
pub fn decodeReader(
    alloc: Allocator,
    reader: *std.Io.Reader,
) error{ EndOfStream, InvalidFileFormat, OutOfMemory, ReadFailed }!Qoi {
    const header = try reader.takeArray(QOI_HEADER_SIZE);

    if (!std.mem.eql(u8, header[0..4], QOI_MAGIC)) return error.InvalidFileFormat;

    const channels: Channels = switch (header[12]) {
        0b011 => .RGB,
        0b100 => .RGBA,
        else => return error.InvalidFileFormat,
    };

    const colorspace: ColorSpace = switch (header[13]) {
        0b0 => .SRGB,
        0b1 => .LINEAR,
        else => return error.InvalidFileFormat,
    };

    var image: Qoi = undefined;
    image.width = btn(header[4..8].*);
    image.height = btn(header[8..12].*);
    image.channels = channels;
    image.colorspace = colorspace;
    image.pixels = try alloc.alloc(rgba, @as(usize, image.width) * @as(usize, image.height));
    errdefer alloc.free(image.pixels);

    var color_lut: [64]rgba = @splat(@splat(0));
    var pix = rgba{ 0xff, 0x00, 0x00, 0x00 };
    var run: usize = 0;

    for (image.pixels) |*pixel| {
        if (run > 0) {
            run -= 1;
        } else {
            const byte = reader.takeByte() catch |err| {
                if (err == error.EndOfStream) break else return err;
            };
            if (byte == QOI_OP_RGB) {
                const data = try reader.takeArray(3);
                pix[0] = data[0];
                pix[1] = data[1];
                pix[2] = data[2];
            } else if (byte == QOI_OP_RGBA) {
                const data = try reader.takeArray(4);
                pix = @as(rgba, @bitCast(data.*));
            } else if (byte & 0xc0 == QOI_OP_INDEX) {
                pix = color_lut[byte];
            } else if (byte & 0xc0 == QOI_OP_DIFF) {
                var diff: rgba = @splat(byte);
                diff >>= rgba{ 4, 2, 0, 0 };
                diff &= @splat(3);
                diff -%= @splat(2);
                diff[3] = 0;
                pix +%= diff;
            } else if (byte & 0xc0 == QOI_OP_LUMA) {
                const data = try reader.takeByte();
                const dg = (byte & 63) -% 32;
                pix +%= rgba{
                    (data >> 4) -% 8 +% dg,
                    dg,
                    (data & 15) -% 8 +% dg,
                    0,
                };
            } else if (byte & 0xc0 == QOI_OP_RUN) {
                run = byte & 0x3f;
            }

            color_lut[hash(pix)] = pix;
        }

        pixel.* = pix;
    }

    return image;
}

/// Allocates a slice to put the encoded data.
///
/// Returns the full allocated slice data along with the length of the encoded slice.
///
/// `inital_capacity`: Optionally pass a size to preallocate for the data arraylist.
pub fn encode(
    self: Qoi,
    alloc: Allocator,
    inital_capacity: ?usize,
) error{ EndOfStream, InvalidFileFormat, OutOfMemory }!std.ArrayList(u8) {
    // Allocates a big ol slice to decode the buffer

    // If they don't pass in the capacity I just use this ratio: https://qoiformat.org/benchmark/ ~ 33%
    const capacity = inital_capacity orelse @divFloor(self.pixels.len, 3);
    var buf = try std.ArrayList(u8).initCapacity(alloc, capacity);
    errdefer buf.deinit(alloc);

    try buf.appendSlice(alloc, QOI_MAGIC ++
        ntb(self.width) ++
        ntb(self.height) ++
        [_]u8{
            @intFromEnum(self.channels),
            @intFromEnum(self.colorspace),
        });

    var color_lut: [64]rgba = @splat(@splat(0));
    var prev = rgba{ 0xff, 0x00, 0x00, 0x00 };
    var run: u8 = 0;

    for (self.pixels, 0..) |pix, i| {
        defer prev = pix;

        const same_pixel = std.meta.eql(pix, prev);

        if (same_pixel) run += 1;

        if (run > 0 and (run == 62 or !same_pixel or (i == (self.pixels.len - 1)))) {
            // QOI_OP_RUN
            std.debug.assert(run >= 1 and run <= 62);
            try buf.append(alloc, QOI_OP_RUN | (run - 1));
            run = 0;
        }

        if (!same_pixel) {
            const pix_hash = hash(pix);
            if (std.meta.eql(color_lut[pix_hash], pix)) {
                try buf.append(alloc, QOI_OP_INDEX | pix_hash);
            } else {
                color_lut[pix_hash] = pix;

                const diff_r, const diff_g, const diff_b, const diff_a =
                    @as(@Vector(4, i16), pix) - @as(@Vector(4, i16), prev);

                const diff_rg = diff_r - diff_g;
                const diff_bg = diff_b - diff_g;

                if (diff_a != 0) {
                    const pixel = [_]u8{ QOI_OP_RGBA, pix[0], pix[1], pix[2], pix[3] };
                    try buf.appendSlice(alloc, &pixel);
                } else if (inRange(i2, diff_r) and inRange(i2, diff_g) and inRange(i2, diff_b)) {
                    try buf.append(
                        alloc,
                        QOI_OP_DIFF |
                            (map2(diff_r) << 4) |
                            (map2(diff_g) << 2) |
                            (map2(diff_b) << 0),
                    );
                } else if (inRange(i6, diff_g) and
                    inRange(i4, diff_rg) and
                    inRange(i4, diff_bg))
                {
                    try buf.appendSlice(alloc, &[_]u8{
                        QOI_OP_LUMA | map6(diff_g),
                        (map4(diff_rg) << 4) | (map4(diff_bg) << 0),
                    });
                } else {
                    const pixel = [_]u8{ QOI_OP_RGB, pix[0], pix[1], pix[2] };
                    try buf.appendSlice(alloc, &pixel);
                }
            }
        }
    }

    try buf.appendSlice(alloc, QOI_EOF);

    return buf;
}

/// Writes the encoded file to the `std.io.Writer`.
pub fn encodeWriter(
    self: Qoi,
    writer: *std.Io.Writer,
) error{WriteFailed}!void {
    try writer.writeAll(QOI_MAGIC ++
        ntb(self.width) ++
        ntb(self.height) ++
        [_]u8{
            @intFromEnum(self.channels),
            @intFromEnum(self.colorspace),
        });

    var color_lut: [64]rgba = @splat(@splat(0));
    var prev = rgba{ 0xff, 0x00, 0x00, 0x00 };
    var run: u8 = 0;

    for (self.pixels, 0..) |pix, i| {
        defer prev = pix;

        const same_pixel = std.meta.eql(pix, prev);

        if (same_pixel) run += 1;

        if (run > 0 and (run == 62 or !same_pixel or (i == (self.pixels.len - 1)))) {
            // QOI_OP_RUN
            std.debug.assert(run >= 1 and run <= 62);
            try writer.writeByte(QOI_OP_RUN | (run - 1));
            run = 0;
        }

        if (!same_pixel) {
            const pix_hash = hash(pix);
            if (std.meta.eql(color_lut[pix_hash], pix)) {
                try writer.writeByte(QOI_OP_INDEX | pix_hash);
            } else {
                color_lut[pix_hash] = pix;

                const diff_r, const diff_g, const diff_b, const diff_a =
                    @as(@Vector(4, i16), pix) - @as(@Vector(4, i16), prev);

                const diff_rg = diff_r - diff_g;
                const diff_rb = diff_b - diff_g;

                if (diff_a != 0) {
                    const pixel = [_]u8{ QOI_OP_RGBA, pix[0], pix[1], pix[2], pix[3] };
                    try writer.writeAll(&pixel);
                } else if (inRange(i2, diff_r) and inRange(i2, diff_g) and inRange(i2, diff_b)) {
                    const byte =
                        QOI_OP_DIFF |
                        (map2(diff_r) << 4) |
                        (map2(diff_g) << 2) |
                        (map2(diff_b) << 0);
                    try writer.writeByte(byte);
                } else if (inRange(i6, diff_g) and
                    inRange(i4, diff_rg) and
                    inRange(i4, diff_rb))
                {
                    try writer.writeByte(QOI_OP_LUMA | map6(diff_g));
                    try writer.writeByte((map4(diff_rg) << 4) | (map4(diff_rb) << 0));
                } else {
                    const pixel = [_]u8{ QOI_OP_RGB, pix[0], pix[1], pix[2] };
                    try writer.writeAll(&pixel);
                }
            }
        }
    }

    try writer.writeAll(QOI_EOF);
    try writer.flush();
}

inline fn ntb(v: u32) [4]u8 {
    return @bitCast(std.mem.nativeToBig(u32, v));
}

inline fn btn(v: [4]u8) u32 {
    return @bitCast(std.mem.bigToNative(u32, @bitCast(v)));
}

inline fn inRange(comptime T: type, val: i16) bool {
    if (@typeInfo(T).int.signedness != .signed) @compileError("");
    return std.math.minInt(T) <= val and val <= std.math.maxInt(T);
}

inline fn map2(val: i16) u8 {
    return @as(u2, @intCast(val + 2));
}

inline fn map4(val: i16) u8 {
    return @as(u4, @intCast(val + 8));
}

inline fn map6(val: i16) u8 {
    return @as(u6, @intCast(val + 32));
}

test "encode decode encode" {
    var rand = std.Random.DefaultPrng.init(0);
    var alloc = std.testing.allocator;

    for (0..100) |seed| {
        {
            rand.seed(seed);
            const rng = rand.random();

            const width = rng.intRangeAtMost(u32, 100, 1000);
            const height = rng.intRangeAtMost(u32, 100, 1000);
            var image = Qoi{
                .width = width,
                .height = height,
                .channels = rng.enumValue(Channels),
                .colorspace = rng.enumValue(ColorSpace),
                .pixels = try alloc.alloc(rgba, @as(usize, width) * @as(usize, height)),
            };
            defer image.deinit(alloc);

            var encoded = try encode(image, alloc, null);
            defer encoded.deinit(alloc);
            var decoded_image = try decode(alloc, encoded.items);
            defer decoded_image.deinit(alloc);

            for (0.., decoded_image.pixels, image.pixels) |idx, l, r| {
                if (@reduce(.And, l != r)) {
                    std.debug.panic("{} l==r failed {any} {any}", .{ idx, l, r });
                }
            }
        }

        {
            rand.seed(seed);
            const rng = rand.random();

            const width = rng.intRangeAtMost(u32, 100, 1000);
            const height = rng.intRangeAtMost(u32, 100, 1000);
            var image = Qoi{
                .width = width,
                .height = height,
                .channels = rng.enumValue(Channels),
                .colorspace = rng.enumValue(ColorSpace),
                .pixels = try alloc.alloc(rgba, @as(usize, width) * @as(usize, height)),
            };
            defer image.deinit(alloc);

            var allocwriter = std.Io.Writer.Allocating.init(alloc);
            defer allocwriter.deinit();

            try encodeWriter(image, &allocwriter.writer);

            var reader = std.Io.Reader.fixed(allocwriter.written());
            var decoded_image = try decodeReader(alloc, &reader);
            defer decoded_image.deinit(alloc);

            for (0.., decoded_image.pixels, image.pixels) |idx, l, r| {
                if (@reduce(.And, l != r)) {
                    std.debug.panic("{} l==r failed {any} {any}", .{ idx, l, r });
                }
            }
        }
    }
}

test "fuzz test failing" {
    var rand = std.Random.DefaultPrng.init(0);
    var alloc = std.testing.allocator;

    for (0..1000) |seed| {
        rand.seed(seed);

        const len = rand.random().intRangeAtMost(usize, 0, 1_000_000);
        const bytes = try alloc.alloc(u8, len);
        defer alloc.free(bytes);

        std.debug.assert(@typeInfo(@TypeOf(decode(alloc, bytes))) == .error_union);
        var reader = std.Io.Reader.fixed(bytes);
        std.debug.assert(@typeInfo(@TypeOf(decodeReader(alloc, &reader))) == .error_union);
    }
}

test "fuzz test success" {
    var rand = std.Random.DefaultPrng.init(0);
    var alloc = std.testing.allocator;

    for (0..100) |seed| {
        {
            rand.seed(seed);
            const rng = rand.random();

            const len = rng.intRangeAtMost(usize, QOI_HEADER_SIZE + QOI_EOF.len, 10_000);
            const bytes = try alloc.alloc(u8, len);
            defer alloc.free(bytes);

            const width = rng.intRangeAtMost(u32, 0, std.math.maxInt(u8));
            const height = rng.intRangeAtMost(u32, 0, std.math.maxInt(u8));
            const channels = rng.enumValue(Channels);
            const colorspace = rng.enumValue(ColorSpace);

            @memcpy(bytes[0..QOI_HEADER_SIZE], QOI_MAGIC ++
                ntb(width) ++
                ntb(height) ++
                [_]u8{
                    @intFromEnum(channels),
                    @intFromEnum(colorspace),
                });

            @memcpy(bytes[bytes.len - QOI_EOF.len ..], QOI_EOF);

            var image = decode(alloc, bytes) catch {
                std.debug.print("seed failed: {}\n", .{seed});
                unreachable;
            };
            defer image.deinit(alloc);
        }

        // for reader
        {
            rand.seed(seed);
            const rng = rand.random();

            const len = rng.intRangeAtMost(usize, QOI_HEADER_SIZE + QOI_EOF.len, 10_000);
            const bytes = try alloc.alloc(u8, len);
            defer alloc.free(bytes);

            const width = rng.intRangeAtMost(u32, 0, std.math.maxInt(u8));
            const height = rng.intRangeAtMost(u32, 0, std.math.maxInt(u8));
            const channels = rng.enumValue(Channels);
            const colorspace = rng.enumValue(ColorSpace);

            @memcpy(bytes[0..QOI_HEADER_SIZE], QOI_MAGIC ++
                ntb(width) ++
                ntb(height) ++
                [_]u8{
                    @intFromEnum(channels),
                    @intFromEnum(colorspace),
                });

            @memcpy(bytes[bytes.len - QOI_EOF.len ..], QOI_EOF);

            var reader = std.io.Reader.fixed(bytes);
            var image = decodeReader(alloc, &reader) catch {
                std.debug.print("seed failed: {}\n", .{seed});
                unreachable;
            };
            defer image.deinit(alloc);
        }
    }
}
