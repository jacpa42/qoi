const std = @import("std");
const Allocator = std.mem.Allocator;

/// A parsed QOI image. Does not hold onto an allocator. Use `image.deinit(alloc)` to free memory.
pub const QOI = struct {
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
pub const rgba = packed struct(u32) {
    r: u8 = 0,
    g: u8 = 0,
    b: u8 = 0,
    a: u8 = 0,

    inline fn hash(c: @This()) u8 {
        return 0x3f & (c.r *% 3 +% c.g *% 5 +% c.b *% 7 +% c.a *% 11);
    }
};

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
}!QOI {
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

    var image = QOI{
        .width = btn(raw_bytes[4..8].*),
        .height = btn(raw_bytes[8..12].*),
        .channels = channels,
        .colorspace = colorspace,
        .pixels = undefined,
    };

    const num_pixels: usize = @as(usize, image.width) * @as(usize, image.height);

    image.pixels = try alloc.alloc(rgba, num_pixels);
    errdefer alloc.free(image.pixels);

    var color_lut = [_]rgba{.{ .a = 0 }} ** 64;
    var pix = rgba{ .a = 0xff };
    var data = raw_bytes[QOI_HEADER_SIZE..];
    var run: usize = 0;

    for (image.pixels) |*pixel| {
        if (run > 0) {
            run -= 1;
        } else if (data.len > QOI_EOF.len) {
            const byte = data[0];
            data = data[1..];

            if (byte == QOI_OP_RGB) {
                pix.r = data[0];
                pix.g = data[1];
                pix.b = data[2];
                data = data[3..];
            } else if (byte == QOI_OP_RGBA) {
                pix = @as(rgba, @bitCast(data[0..4].*));
                data = data[4..];
            } else if (byte & 0xc0 == QOI_OP_INDEX) {
                pix = color_lut[byte];
            } else if (byte & 0xc0 == QOI_OP_DIFF) {
                pix.r +%= (0b11 & (byte >> 4)) -% 2;
                pix.g +%= (0b11 & (byte >> 2)) -% 2;
                pix.b +%= (0b11 & (byte >> 0)) -% 2;
            } else if (byte & 0xc0 == QOI_OP_LUMA) {
                const dg = (byte & 63) -% 32;
                pix.r +%= (data[0] >> 4) -% 8 +% dg;
                pix.g +%= dg;
                pix.b +%= (data[0] & 15) -% 8 +% dg;
                data = data[1..];
            } else if (byte & 0xc0 == QOI_OP_RUN) {
                run = byte & 0x3f;
            }

            color_lut[pix.hash()] = pix;
        }

        pixel.* = pix;
    }

    return image;
}

/// Parses bytes from reader into `QOI`.
pub fn decodeReader(
    alloc: Allocator,
    reader: *std.Io.Reader,
) error{ EndOfStream, InvalidFileFormat, OutOfMemory, ReadFailed }!QOI {
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

    var image = QOI{
        .width = btn(header[4..8].*),
        .height = btn(header[8..12].*),
        .channels = channels,
        .colorspace = colorspace,
        .pixels = undefined,
    };

    const num_pixels: usize = @as(usize, image.width) * @as(usize, image.height);

    image.pixels = try alloc.alloc(rgba, num_pixels);
    errdefer alloc.free(image.pixels);

    var color_lut = [_]rgba{.{ .a = 0 }} ** 64;
    var pix = rgba{ .a = 0xff };
    var run: usize = 0;

    for (image.pixels) |*pixel| {
        if (run > 0) {
            run -= 1;
        } else {
            const byte = try reader.takeByte();

            if (byte == QOI_OP_RGB) {
                const data = try reader.takeArray(3);
                pix.r = data[0];
                pix.g = data[1];
                pix.b = data[2];
            } else if (byte == QOI_OP_RGBA) {
                const data = try reader.takeArray(4);
                pix = @as(rgba, @bitCast(data.*));
            } else if (byte & 0xc0 == QOI_OP_INDEX) {
                pix = color_lut[byte];
            } else if (byte & 0xc0 == QOI_OP_DIFF) {
                pix.r +%= (0b11 & (byte >> 4)) -% 2;
                pix.g +%= (0b11 & (byte >> 2)) -% 2;
                pix.b +%= (0b11 & (byte >> 0)) -% 2;
            } else if (byte & 0xc0 == QOI_OP_LUMA) {
                const data = try reader.takeByte();
                const dg = (byte & 63) -% 32;
                pix.r +%= (data >> 4) -% 8 +% dg;
                pix.g +%= dg;
                pix.b +%= (data & 15) -% 8 +% dg;
            } else if (byte & 0xc0 == QOI_OP_RUN) {
                run = byte & 0x3f;
            }

            color_lut[pix.hash()] = pix;
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
    self: *const QOI,
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

    var color_lut = [_]rgba{.{ .a = 0 }} ** 64;
    var prev = rgba{ .a = 0xff };
    var run_length: u8 = 0;

    for (self.pixels, 0..) |pix, i| {
        defer prev = pix;

        const same_pixel = std.meta.eql(pix, prev);

        if (same_pixel) run_length += 1;

        if (run_length > 0 and (run_length == 62 or !same_pixel or (i == (self.pixels.len - 1)))) {
            // QOI_OP_RUN
            std.debug.assert(run_length >= 1 and run_length <= 62);
            try buf.append(alloc, QOI_OP_RUN | (run_length - 1));
            run_length = 0;
        }

        if (!same_pixel) {
            const pix_hash = pix.hash();
            if (std.meta.eql(color_lut[pix_hash], pix)) {
                try buf.append(alloc, QOI_OP_INDEX | pix_hash);
            } else {
                color_lut[pix_hash] = pix;

                const diff_r = @as(i16, pix.r) - @as(i16, prev.r);
                const diff_g = @as(i16, pix.g) - @as(i16, prev.g);
                const diff_b = @as(i16, pix.b) - @as(i16, prev.b);
                const diff_a = @as(i16, pix.a) - @as(i16, prev.a);

                const diff_rg = diff_r - diff_g;
                const diff_rb = diff_b - diff_g;

                if (diff_a != 0) {
                    const pixel = [_]u8{ QOI_OP_RGBA, pix.r, pix.g, pix.b, pix.a };
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
                    inRange(i4, diff_rb))
                {
                    try buf.appendSlice(alloc, &[_]u8{
                        QOI_OP_LUMA | map6(diff_g),
                        (map4(diff_rg) << 4) | (map4(diff_rb) << 0),
                    });
                } else {
                    const pixel = [_]u8{ QOI_OP_RGB, pix.r, pix.g, pix.b };
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
    self: *const QOI,
    writer: *std.Io.Writer,
) error{WriteFailed}!void {
    try writer.writeAll(QOI_MAGIC ++
        ntb(self.width) ++
        ntb(self.height) ++
        [_]u8{
            @intFromEnum(self.channels),
            @intFromEnum(self.colorspace),
        });

    var color_lut = [_]rgba{.{ .a = 0 }} ** 64;
    var prev = rgba{ .a = 0xff };
    var run_length: u8 = 0;

    for (self.pixels, 0..) |pix, i| {
        defer prev = pix;

        const same_pixel = std.meta.eql(pix, prev);

        if (same_pixel) run_length += 1;

        if (run_length > 0 and (run_length == 62 or !same_pixel or (i == (self.pixels.len - 1)))) {
            // QOI_OP_RUN
            std.debug.assert(run_length >= 1 and run_length <= 62);
            try writer.writeByte(QOI_OP_RUN | (run_length - 1));
            run_length = 0;
        }

        if (!same_pixel) {
            const pix_hash = pix.hash();
            if (std.meta.eql(color_lut[pix_hash], pix)) {
                try writer.writeByte(QOI_OP_INDEX | pix_hash);
            } else {
                color_lut[pix_hash] = pix;

                const diff_r = @as(i16, pix.r) - @as(i16, prev.r);
                const diff_g = @as(i16, pix.g) - @as(i16, prev.g);
                const diff_b = @as(i16, pix.b) - @as(i16, prev.b);
                const diff_a = @as(i16, pix.a) - @as(i16, prev.a);

                const diff_rg = diff_r - diff_g;
                const diff_rb = diff_b - diff_g;

                if (diff_a != 0) {
                    const pixel = [_]u8{ QOI_OP_RGBA, pix.r, pix.g, pix.b, pix.a };
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
                    const buf = [_]u8{
                        QOI_OP_LUMA | map6(diff_g),
                        (map4(diff_rg) << 4) | (map4(diff_rb) << 0),
                    };
                    try writer.writeAll(buf);
                } else {
                    const pixel = [_]u8{ QOI_OP_RGB, pix.r, pix.g, pix.b };
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

test "fuzz test failing" {
    var rand = std.Random.DefaultPrng.init(0);
    var alloc = std.testing.allocator;

    for (0..1000) |seed| {
        rand.seed(seed);

        const len = rand.random().intRangeAtMost(usize, 0, 1_000_000);
        const bytes = try alloc.alloc(u8, len);
        defer alloc.free(bytes);

        std.debug.assert(@typeInfo(@TypeOf(decode(alloc, bytes))) == .error_union);
    }
}

test "fuzz test success" {
    var rand = std.Random.DefaultPrng.init(0);
    var alloc = std.testing.allocator;

    for (0..100) |seed| {
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
}
