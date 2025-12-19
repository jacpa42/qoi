const std = @import("std");
const Allocator = std.mem.Allocator;

pub const DecodeError = error{
    InvalidFileFormat,
    OutOfMemory,
    InvalidNumberOfChannels,
    InvalidColorSpaceDescription,
};
pub const rgba = @Vector(4, u8);
pub const Channels = enum(u8) { RGB = 3, RGBA = 4 };
pub const ColorSpace = enum(u8) { SRGB = 0, LINEAR = 1 };
pub const OP = struct {
    const RGB = 0xfe;
    const RGBA = 0xff;
    const INDEX = 0x00;
    const DIFF = 0x40;
    const LUMA = 0x80;
    const RUN = 0xc0;
};

const MAGIC = "qoif";
const EOF = "\x00" ** 7 ++ "\x01";
const HEADER_SIZE = 14;

/// A parsed QOI image. Does not hold onto an allocator. Use `image.deinit(alloc)` to free memory.
const QOI = @This();

width: u32,
height: u32,
channels: Channels,
colorspace: ColorSpace,

/// Pixel array list. Memory is externally managed.
pixels: []rgba,

pub fn deinit(self: QOI, alloc: Allocator) void {
    alloc.free(self.pixels);
}

/// Parses bytes from the buffer into `QOI`.
///
/// Needs allocator to put the pixel data. Use `deinit(alloc)` do deinitialize.
pub fn decode(
    alloc: Allocator,
    raw_bytes: []const u8,
) DecodeError!QOI {
    if (raw_bytes.len <= HEADER_SIZE or
        !std.mem.eql(u8, raw_bytes[0..4], MAGIC))
    {
        return error.InvalidFileFormat;
    }

    var image: QOI = undefined;
    image.width = btn(raw_bytes[4..8].*);
    image.height = btn(raw_bytes[8..12].*);
    image.channels = ite(Channels, raw_bytes[12]) catch return error.InvalidNumberOfChannels;
    image.colorspace = ite(ColorSpace, raw_bytes[13]) catch return error.InvalidNumberOfChannels;
    image.pixels = try alloc.alloc(rgba, @as(usize, image.width) * @as(usize, image.height));
    errdefer alloc.free(image.pixels);

    var color_lut: [64]rgba = @splat(@splat(0));
    var pix = rgba{ 0x00, 0x00, 0x00, 0xff };
    var data = raw_bytes[HEADER_SIZE..];
    var run: usize = 0;

    for (image.pixels) |*pixel| {
        if (run > 0) {
            run -= 1;
        } else if (data.len > EOF.len) {
            const byte = data[0];
            data = data[1..];

            if (byte == OP.RGB) {
                pix[0] = data[0];
                pix[1] = data[1];
                pix[2] = data[2];
                data = data[3..];
            } else if (byte == OP.RGBA) {
                pix = data[0..4].*;
                data = data[4..];
            } else if (byte & 0xc0 == OP.INDEX) {
                pix = color_lut[byte];
            } else if (byte & 0xc0 == OP.DIFF) {
                var diff: rgba = @splat(byte);
                diff >>= rgba{ 4, 2, 0, 0 };
                diff &= @splat(3);
                diff -%= @splat(2);
                diff[3] = 0;
                pix +%= diff;
            } else if (byte & 0xc0 == OP.LUMA) {
                const dg = (byte & 63) -% 32;
                pix +%= rgba{
                    (data[0] >> 4) -% 8 +% dg,
                    dg,
                    (data[0] & 15) -% 8 +% dg,
                    0,
                };
                data = data[1..];
            } else if (byte & 0xc0 == OP.RUN) {
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
) (DecodeError || error{ReadFailed})!QOI {
    const header = reader.takeArray(HEADER_SIZE) catch {
        return error.InvalidFileFormat;
    };

    if (!std.mem.eql(u8, header[0..4], MAGIC)) return error.InvalidFileFormat;

    var image: QOI = undefined;
    image.width = btn(header[4..8].*);
    image.height = btn(header[8..12].*);
    image.channels = ite(Channels, header[12]) catch return error.InvalidNumberOfChannels;
    image.colorspace = ite(ColorSpace, header[13]) catch return error.InvalidColorSpaceDescription;
    image.pixels = try alloc.alloc(rgba, @as(usize, image.width) * @as(usize, image.height));
    errdefer alloc.free(image.pixels);

    var color_lut: [64]rgba = @splat(@splat(0));
    var pix = rgba{ 0x00, 0x00, 0x00, 0xff };
    var run: usize = 0;

    for (image.pixels) |*pixel| {
        if (run > 0) {
            run -= 1;
        } else {
            _ = reader.peek(EOF.len + 1) catch |e| {
                if (e == error.EndOfStream) break else return error.InvalidFileFormat;
            };
            const byte = reader.takeByte() catch unreachable;
            if (byte == OP.RGB) {
                const data = reader.takeArray(3) catch unreachable;
                pix[0] = data[0];
                pix[1] = data[1];
                pix[2] = data[2];
            } else if (byte == OP.RGBA) {
                const data = reader.takeArray(4) catch unreachable;
                pix = @as(rgba, @bitCast(data.*));
            } else if (byte & 0xc0 == OP.INDEX) {
                pix = color_lut[byte];
            } else if (byte & 0xc0 == OP.DIFF) {
                var diff: rgba = @splat(byte);
                diff >>= rgba{ 4, 2, 0, 0 };
                diff &= @splat(3);
                diff -%= @splat(2);
                diff[3] = 0;
                pix +%= diff;
            } else if (byte & 0xc0 == OP.LUMA) {
                const data = reader.takeByte() catch unreachable;
                const dg = (byte & 0x3f) -% 32;
                pix +%= rgba{
                    (data >> 4) -% 8 +% dg,
                    dg,
                    (data & 15) -% 8 +% dg,
                    0,
                };
            } else if (byte & 0xc0 == OP.RUN) {
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
    self: QOI,
    alloc: Allocator,
) error{ EndOfStream, InvalidFileFormat, OutOfMemory }!std.ArrayList(u8) {
    // Allocates a big ol slice to decode the buffer
    // I just use this ratio: https://qoiformat.org/benchmark/ ~ 33%
    var buf = try std.ArrayList(u8).initCapacity(alloc, @divFloor(self.pixels.len, 3));
    errdefer buf.deinit(alloc);

    try buf.appendSlice(alloc, MAGIC ++
        ntb(self.width) ++
        ntb(self.height) ++
        [_]u8{
            @intFromEnum(self.channels),
            @intFromEnum(self.colorspace),
        });

    var color_lut: [64]rgba = @splat(@splat(0));
    var prev = rgba{ 0x00, 0x00, 0x00, 0xff };
    var run: u8 = 0;

    for (self.pixels, 0..) |pix, i| {
        defer prev = pix;

        const same_pixel = std.meta.eql(pix, prev);

        if (same_pixel) run += 1;

        if (run > 0 and (run == 62 or !same_pixel or (i == (self.pixels.len - 1)))) {
            // Op.RUN
            std.debug.assert(run >= 1 and run <= 62);
            try buf.append(alloc, OP.RUN | (run - 1));
            run = 0;
        }

        if (!same_pixel) {
            const pix_hash = hash(pix);
            if (std.meta.eql(color_lut[pix_hash], pix)) {
                try buf.append(alloc, OP.INDEX | pix_hash);
            } else {
                color_lut[pix_hash] = pix;

                const diff_r, const diff_g, const diff_b, const diff_a =
                    @as(@Vector(4, i16), pix) - @as(@Vector(4, i16), prev);

                const diff_rg = diff_r - diff_g;
                const diff_bg = diff_b - diff_g;

                if (diff_a != 0) {
                    const pixel = [_]u8{ OP.RGBA, pix[0], pix[1], pix[2], pix[3] };
                    try buf.appendSlice(alloc, &pixel);
                } else if (inRange(i2, diff_r) and inRange(i2, diff_g) and inRange(i2, diff_b)) {
                    try buf.append(
                        alloc,
                        OP.DIFF |
                            (map2(diff_r) << 4) |
                            (map2(diff_g) << 2) |
                            (map2(diff_b) << 0),
                    );
                } else if (inRange(i6, diff_g) and
                    inRange(i4, diff_rg) and
                    inRange(i4, diff_bg))
                {
                    try buf.appendSlice(alloc, &[_]u8{
                        OP.LUMA | map6(diff_g),
                        (map4(diff_rg) << 4) | (map4(diff_bg) << 0),
                    });
                } else {
                    const pixel = [_]u8{ OP.RGB, pix[0], pix[1], pix[2] };
                    try buf.appendSlice(alloc, &pixel);
                }
            }
        }
    }

    try buf.appendSlice(alloc, EOF);

    return buf;
}

/// Writes the encoded file to the `std.io.Writer`.
pub fn encodeWriter(
    self: QOI,
    writer: *std.Io.Writer,
) error{WriteFailed}!void {
    try writer.writeAll(MAGIC ++
        ntb(self.width) ++
        ntb(self.height) ++
        [_]u8{ @intFromEnum(self.channels), @intFromEnum(self.colorspace) });

    var color_lut: [64]rgba = @splat(@splat(0));
    var prev = rgba{ 0x00, 0x00, 0x00, 0xff };
    var run: u8 = 0;

    for (self.pixels, 0..) |pix, i| {
        defer prev = pix;

        const same_pixel = std.meta.eql(pix, prev);

        if (same_pixel) run += 1;

        if (run > 0 and (run == 62 or !same_pixel or (i == (self.pixels.len - 1)))) {
            // Op.RUN
            std.debug.assert(run >= 1 and run <= 62);
            try writer.writeByte(OP.RUN | (run - 1));
            run = 0;
        }

        if (!same_pixel) {
            const pix_hash = hash(pix);
            if (std.meta.eql(color_lut[pix_hash], pix)) {
                try writer.writeByte(OP.INDEX | pix_hash);
            } else {
                color_lut[pix_hash] = pix;

                const diff_r, const diff_g, const diff_b, const diff_a =
                    @as(@Vector(4, i16), pix) - @as(@Vector(4, i16), prev);

                const diff_rg = diff_r - diff_g;
                const diff_rb = diff_b - diff_g;

                if (diff_a != 0) {
                    const pixel = [_]u8{ OP.RGBA, pix[0], pix[1], pix[2], pix[3] };
                    try writer.writeAll(&pixel);
                } else if (inRange(i2, diff_r) and inRange(i2, diff_g) and inRange(i2, diff_b)) {
                    const byte =
                        OP.DIFF |
                        (map2(diff_r) << 4) |
                        (map2(diff_g) << 2) |
                        (map2(diff_b) << 0);
                    try writer.writeByte(byte);
                } else if (inRange(i6, diff_g) and
                    inRange(i4, diff_rg) and
                    inRange(i4, diff_rb))
                {
                    try writer.writeByte(OP.LUMA | map6(diff_g));
                    try writer.writeByte((map4(diff_rg) << 4) | (map4(diff_rb) << 0));
                } else {
                    const pixel = [_]u8{ OP.RGB, pix[0], pix[1], pix[2] };
                    try writer.writeAll(&pixel);
                }
            }
        }
    }

    try writer.writeAll(EOF);
    try writer.flush();
}

const ite = std.meta.intToEnum;

inline fn hash(color: rgba) u8 {
    return 0x3f & @reduce(.Add, color *% rgba{ 3, 5, 7, 11 });
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
            var image = QOI{
                .width = width,
                .height = height,
                .channels = rng.enumValue(Channels),
                .colorspace = rng.enumValue(ColorSpace),
                .pixels = try alloc.alloc(rgba, @as(usize, width) * @as(usize, height)),
            };
            defer image.deinit(alloc);

            var encoded = try encode(image, alloc);
            defer encoded.deinit(alloc);
            var decoded_image = try decode(alloc, encoded.items);
            defer decoded_image.deinit(alloc);

            try std.testing.expectEqualDeep(image, decoded_image);
        }

        {
            rand.seed(seed);
            const rng = rand.random();

            const width = rng.intRangeAtMost(u32, 100, 1000);
            const height = rng.intRangeAtMost(u32, 100, 1000);
            var image = QOI{
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

            try std.testing.expectEqualDeep(image, decoded_image);
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

            const len = rng.intRangeAtMost(usize, HEADER_SIZE + EOF.len, 10_000);
            const bytes = try alloc.alloc(u8, len);
            defer alloc.free(bytes);

            const width = rng.intRangeAtMost(u32, 0, std.math.maxInt(u8));
            const height = rng.intRangeAtMost(u32, 0, std.math.maxInt(u8));
            const channels = rng.enumValue(Channels);
            const colorspace = rng.enumValue(ColorSpace);

            @memcpy(bytes[0..HEADER_SIZE], MAGIC ++
                ntb(width) ++
                ntb(height) ++
                [_]u8{
                    @intFromEnum(channels),
                    @intFromEnum(colorspace),
                });

            @memcpy(bytes[bytes.len - EOF.len ..], EOF);

            var image = try decode(alloc, bytes);
            defer image.deinit(alloc);
        }

        // for reader
        {
            rand.seed(seed);
            const rng = rand.random();

            const len = rng.intRangeAtMost(usize, HEADER_SIZE + EOF.len, 10_000);
            const bytes = try alloc.alloc(u8, len);
            defer alloc.free(bytes);

            const width = rng.intRangeAtMost(u32, 0, std.math.maxInt(u8));
            const height = rng.intRangeAtMost(u32, 0, std.math.maxInt(u8));
            const channels = rng.enumValue(Channels);
            const colorspace = rng.enumValue(ColorSpace);

            @memcpy(bytes[0..HEADER_SIZE], MAGIC ++
                ntb(width) ++
                ntb(height) ++
                [_]u8{
                    @intFromEnum(channels),
                    @intFromEnum(colorspace),
                });

            @memcpy(bytes[bytes.len - EOF.len ..], EOF);

            var reader = std.io.Reader.fixed(bytes);
            var image = try decodeReader(alloc, &reader);
            defer image.deinit(alloc);
        }
    }
}
