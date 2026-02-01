const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Options = struct {
    // Flip the image vertically
    flip: Flip = .none,
    pub const Flip = enum {
        none,
        /// Flip pixels along the x axis
        x,
        /// Flip pixels along the y axis
        y,
        /// Flip pixels along the x and y axis
        xy,
    };
};
pub const DecodeError = error{
    ExpectedMorePixelData,
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
    options: Options,
) DecodeError!QOI {
    var reader = std.io.Reader.fixed(raw_bytes);
    return decodeReader(alloc, &reader, options);
}

/// Decodes and returns the next pixel
fn decodePixelReader(
    pix: *rgba,
    run: *usize,
    color_lut: *[64]rgba,
    reader: *std.io.Reader,
) error{ InvalidFileFormat, ExpectedMorePixelData }!rgba {
    if (run.* > 0) {
        run.* -= 1;
        return pix.*;
    }

    _ = reader.peek(EOF.len + 1) catch |e| {
        if (e == error.EndOfStream) {
            return error.ExpectedMorePixelData;
        } else {
            return error.InvalidFileFormat;
        }
    };

    const byte = reader.takeByte() catch unreachable;
    if (byte == OP.RGB) {
        const data = reader.takeArray(3) catch unreachable;
        pix[0] = data[0];
        pix[1] = data[1];
        pix[2] = data[2];
    } else if (byte == OP.RGBA) {
        const data = reader.takeArray(4) catch unreachable;
        pix.* = @as(rgba, @bitCast(data.*));
    } else if (byte & 0xc0 == OP.INDEX) {
        pix.* = color_lut[byte];
    } else if (byte & 0xc0 == OP.DIFF) {
        var diff: rgba = @splat(byte);
        diff >>= rgba{ 4, 2, 0, 0 };
        diff &= @splat(3);
        diff -%= @splat(2);
        diff[3] = 0;
        pix.* +%= diff;
    } else if (byte & 0xc0 == OP.LUMA) {
        const data = reader.takeByte() catch unreachable;
        const dg = (byte & 0x3f) -% 32;
        pix.* +%= rgba{
            (data >> 4) -% 8 +% dg,
            dg,
            (data & 15) -% 8 +% dg,
            0,
        };
    } else if (byte & 0xc0 == OP.RUN) {
        run.* = byte & 0x3f;
    }

    color_lut[hash(pix.*)] = pix.*;
    return pix.*;
}

/// Parses bytes from reader into `QOI`.
pub fn decodeReader(
    alloc: Allocator,
    reader: *std.Io.Reader,
    options: Options,
) DecodeError!QOI {
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

    const ystart, const ystep, const xstart, const xstep = switch (options.flip) {
        .none => .{ 0, image.width, 0, 1 },
        .x => .{ 0, image.width, image.width -% 1, -%@as(usize, 1) },
        .y => .{ image.pixels.len -% image.width, -%@as(usize, image.width), 0, 1 },
        .xy => .{ image.pixels.len -% image.width, -%@as(usize, image.width), image.width -% 1, -%@as(usize, 1) },
    };

    var y: usize = ystart;
    while (y < image.pixels.len) : (y +%= ystep) {
        var x: usize = xstart;
        while (x < image.width) : (x +%= xstep) {
            image.pixels[y + x] = try decodePixelReader(&pix, &run, &color_lut, reader);
        }
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
    options: Options,
) error{WriteFailed}!std.ArrayList(u8) {
    var out = std.io.Writer.Allocating.init(alloc);
    errdefer out.deinit();

    try encodeWriter(self, &out.writer, options);

    return out.toArrayList();
}

fn encodePixelWriter(
    num_pixels: usize,
    writer: *std.Io.Writer,
    color_lut: *[64]rgba,
    prev: *rgba,
    run: *u8,
    pix: rgba,
    i: usize,
) error{WriteFailed}!void {
    defer prev.* = pix;

    const same_pixel = std.meta.eql(pix, prev.*);

    if (same_pixel) run.* += 1;

    if (run.* > 0 and (run.* == 62 or !same_pixel or (i == (num_pixels - 1)))) {
        // Op.RUN
        std.debug.assert(run.* >= 1 and run.* <= 62);
        try writer.writeByte(OP.RUN | (run.* - 1));
        run.* = 0;
    }

    if (!same_pixel) {
        const pix_hash = hash(pix);
        if (std.meta.eql(color_lut[pix_hash], pix)) {
            try writer.writeByte(OP.INDEX | pix_hash);
        } else {
            color_lut[pix_hash] = pix;

            const diff_r, const diff_g, const diff_b, const diff_a =
                @as(@Vector(4, i16), pix) - @as(@Vector(4, i16), prev.*);

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

/// Writes the encoded file to the `std.io.Writer`.
pub fn encodeWriter(
    self: QOI,
    writer: *std.Io.Writer,
    options: Options,
) error{WriteFailed}!void {
    try writer.writeAll(MAGIC ++
        ntb(self.width) ++
        ntb(self.height) ++
        [_]u8{ @intFromEnum(self.channels), @intFromEnum(self.colorspace) });

    const num_pixels = self.pixels.len;
    var color_lut: [64]rgba = @splat(@splat(0));
    var prev = rgba{ 0x00, 0x00, 0x00, 0xff };
    var run: u8 = 0;

    const ystart, const ystep, const xstart, const xstep = switch (options.flip) {
        .none => .{ 0, self.width, 0, 1 },
        .x => .{ 0, self.width, self.width -% 1, -%@as(usize, 1) },
        .y => .{ self.pixels.len -% self.width, -%@as(usize, self.width), 0, 1 },
        .xy => .{ self.pixels.len -% self.width, -%@as(usize, self.width), self.width -% 1, -%@as(usize, 1) },
    };

    var y: usize = ystart;
    while (y < num_pixels) : (y +%= ystep) {
        var x: usize = xstart;
        while (x < self.width) : (x +%= xstep) {
            const pix = self.pixels[y + x];
            try encodePixelWriter(num_pixels, writer, &color_lut, &prev, &run, pix, y + x);
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

            var encoded = try encode(image, alloc, .{});
            defer encoded.deinit(alloc);
            var decoded_image = try decode(alloc, encoded.items, .{});
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

        std.debug.assert(@typeInfo(@TypeOf(decode(alloc, bytes, .{}))) == .error_union);
    }
}

test "fuzz test success" {
    var rand = std.Random.DefaultPrng.init(0);
    var alloc = std.testing.allocator;

    for (0..100) |seed| {
        {
            rand.seed(seed);
            const rng = rand.random();

            var qoi: QOI = undefined;

            qoi.width = rng.intRangeAtMost(u32, 0, std.math.maxInt(u12));
            qoi.height = rng.intRangeAtMost(u32, 0, std.math.maxInt(u12));
            qoi.channels = rng.enumValue(Channels);
            qoi.colorspace = rng.enumValue(ColorSpace);
            qoi.pixels = try alloc.alloc(rgba, @as(usize, qoi.width * qoi.height));
            defer qoi.deinit(alloc);

            var encoded = try qoi.encode(alloc, .{});
            defer encoded.deinit(alloc);

            var image = try decode(alloc, encoded.items, .{});
            defer image.deinit(alloc);
        }
    }
}
