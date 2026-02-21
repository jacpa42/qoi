const std = @import("std");

pub const DecodeError = error{
    ExpectedMorePixelData,
    InvalidFileFormat,
    OutOfMemory,
    InvalidNumberOfChannels,
    InvalidColorSpaceDescription,
};
pub const irgba = @Vector(4, i8);
pub const rgba = @Vector(4, u8);
pub const Channels = enum(u8) { rgb = 3, rgba = 4 };
pub const ColorSpace = enum(u8) { srgb = 0, linear = 1 };
pub const Op = struct {
    const rgb = 0xfe;
    const rgba = 0xff;
    const index = 0x00;
    const diff = 0x40;
    const luma = 0x80;
    const run = 0xc0;
};

const max_pixels = 400_000_000;
const magic = "qoif";
const eof = "\x00" ** 7 ++ "\x01";
const header_size = 14;

/// A parsed QOI image. Does not hold onto an allocator. Use `image.deinit(alloc)` to free memory.
const QOI = @This();

width: u32,
height: u32,
channels: Channels,
colorspace: ColorSpace,

/// Pixel array list. Memory is externally managed.
pixel_data: []u8,

pub fn deinit(self: QOI, alloc: std.mem.Allocator) void {
    alloc.free(self.pixel_data);
}

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

/// Parses bytes from the buffer into `QOI`.
///
/// Needs allocator to put the pixel data. Use `deinit(alloc)` do deinitialize.
pub fn decode(
    alloc: std.mem.Allocator,
    slice: []const u8,
    options: Options,
) DecodeError!QOI {
    var reader = std.Io.Reader.fixed(slice);
    return decodeReader(alloc, &reader, options);
}

/// Parses bytes from reader into `QOI`.
pub fn decodeReader(
    alloc: std.mem.llocator,
    reader: *std.Io.Reader,
    options: Options,
) DecodeError!QOI {
    const header = reader.takeArray(header_size) catch return error.InvalidFileFormat;
    if (!std.mem.eql(u8, header[0..4], magic)) return error.InvalidFileFormat;

    var image: QOI = undefined;
    image.width = btn(header[4..8].*);
    image.height = btn(header[8..12].*);
    image.channels = ite(Channels, header[12]) catch return error.InvalidNumberOfChannels;
    image.colorspace = ite(ColorSpace, header[13]) catch return error.InvalidColorSpaceDescription;

    const pixel_size: usize = @intFromEnum(image.channels);
    const num_pixels = @as(usize, image.width) * @as(usize, image.height);

    if (num_pixels > max_pixels) return error.OutOfMemory;
    image.pixel_data = try alloc.alloc(u8, pixel_size * num_pixels);
    errdefer alloc.free(image.pixel_data);

    var color_lut: [64]rgba = @splat(@splat(0));
    var pixel = rgba{ 0x00, 0x00, 0x00, 0xff };
    var run: usize = 0;

    var ystart: usize = 0;
    var ystep: usize = image.width * pixel_size;
    var xstart: usize = 0;
    var xstep: usize = pixel_size;
    switch (options.flip) {
        .none => {},
        .x => {
            xstart = (image.width -% 1) * pixel_size;
            xstep = -%xstep;
        },
        .y => {
            ystart = (num_pixels -% image.width) * pixel_size;
            ystep = -%ystep;
        },
        .xy => {
            ystart = (num_pixels -% image.width) * pixel_size;
            ystep = -%ystep;
            xstart = (image.width -% 1) * pixel_size;
            xstep = -%xstep;
        },
    }

    var y: usize = ystart;
    while (y < num_pixels * pixel_size) : (y +%= ystep) {
        var x: usize = xstart;
        while (x < image.width * pixel_size) : (x +%= xstep) {
            if (run > 0) {
                @branchHint(.unlikely);
                run -= 1;
            } else {
                @branchHint(.likely);
                _ = reader.peek(eof.len + 1) catch |e| switch (e) {
                    error.EndOfStream => return error.ExpectedMorePixelData,
                    error.ReadFailed => return error.InvalidFileFormat,
                };
                defer color_lut[hash(pixel)] = pixel;

                const byte = reader.takeByte() catch unreachable;
                switch (byte >> 6) {
                    Op.index >> 6 => pixel = color_lut[byte],
                    Op.diff >> 6 => {
                        var diff: rgba = @splat(byte);
                        diff >>= rgba{ 4, 2, 0, 0 };
                        diff &= @splat(3);
                        diff -%= @splat(2);
                        diff[3] = 0;
                        pixel +%= diff;
                    },
                    Op.luma >> 6 => {
                        const data = reader.takeByte() catch unreachable;
                        const dg = (byte & 63) -% 32;
                        pixel +%= rgba{
                            (data >> 4) -% 8 +% dg,
                            dg,
                            (data & 15) -% 8 +% dg,
                            0,
                        };
                    },
                    Op.run >> 6 => switch (byte) {
                        Op.rgb => {
                            const data = reader.takeArray(3) catch unreachable;
                            pixel[0] = data[0];
                            pixel[1] = data[1];
                            pixel[2] = data[2];
                        },
                        Op.rgba => {
                            pixel = @as(rgba, (reader.takeArray(4) catch unreachable).*);
                        },
                        else => run = byte & 0x3f, // run
                    },
                    else => unreachable,
                }
            }

            image.pixel_data[y + x + 0] = pixel[0];
            image.pixel_data[y + x + 1] = pixel[1];
            image.pixel_data[y + x + 2] = pixel[2];
            if (image.channels == .rgba) {
                image.pixel_data[y + x + 3] = pixel[3];
            }
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
    alloc: std.mem.Allocator,
) error{WriteFailed}!std.ArrayList(u8) {
    var out = std.io.Writer.Allocating.init(alloc);
    errdefer out.deinit();
    try encodeWriter(self, &out.writer);
    return out.toArrayList();
}

/// Writes the encoded file to the `std.io.Writer`.
pub fn encodeWriter(
    self: QOI,
    writer: *std.Io.Writer,
) error{WriteFailed}!void {
    const pixel_info = [2]u8{ @intFromEnum(self.channels), @intFromEnum(self.colorspace) };
    const header = magic ++ ntb(self.width) ++ ntb(self.height) ++ pixel_info;
    try writer.writeAll(header);

    const pixel_size: usize = @intFromEnum(self.channels);
    const num_pixels = @as(usize, self.width) * @as(usize, self.height);

    if (num_pixels > max_pixels) return error.WriteFailed;

    var color_lut: [64]rgba = @splat(@splat(0));
    var prev = rgba{ 0x00, 0x00, 0x00, 0xff };
    var run: u8 = 0;

    var i: usize = 0;
    while (i < self.pixel_data.len) : (i += pixel_size) {
        const pix = rgba{
            self.pixel_data[i + 0],
            self.pixel_data[i + 1],
            self.pixel_data[i + 2],
            if (self.channels == .rgba) self.pixel_data[i + 3] else 0xff,
        };
        defer prev = pix;

        if (@reduce(.And, pix == prev)) {
            run += 1;
            if (run == 62 or i == self.pixel_data.len - pixel_size) {
                std.debug.assert(run >= 1 and run <= 62);
                try writer.writeByte(Op.run | (run - 1));
                run = 0;
            }
        } else {
            if (run > 0) {
                try writer.writeByte(Op.run | (run - 1));
                run = 0;
            }

            const pos = hash(pix);
            if (@reduce(.And, color_lut[pos] == pix)) {
                try writer.writeByte(Op.index | pos);
            } else {
                color_lut[pos] = pix;

                if ((pix == prev)[3]) {
                    // alpha element is 0!
                    var diff: irgba = @bitCast(pix -% prev);
                    std.debug.assert(diff[3] == 0);
                    var diff2 = diff;
                    diff2 -%= irgba{ diff[1], 0, diff[1], 0 };

                    if (checkDiff(diff)) {
                        diff += @splat(2);
                        const v: u8 = @reduce(.Or, @as(rgba, @bitCast(diff)) * rgba{ 0x10, 0x04, 0x01, 0x00 });
                        try writer.writeByte(Op.diff | v);
                    } else if (checkLuma(diff2)) {
                        try writer.writeAll(&.{ Op.luma | m6(diff[1]), m4(diff2[0]) << 4 | m4(diff2[2]) });
                    } else {
                        try writer.writeAll(&([_]u8{Op.rgb} ++ @as([4]u8, pix)[0..3].*));
                    }
                } else {
                    try writer.writeAll(&([_]u8{Op.rgba} ++ @as([4]u8, pix)));
                }
            }
        }
    }

    try writer.writeAll(eof);
    try writer.flush();
}

const ite = std.meta.intToEnum;

// zig fmt: off
fn hash(color: rgba) u6 { return @truncate(@reduce(.Add, color *% rgba{ 3, 5, 7, 11 })); }
fn ntb(v: u32) [4]u8 { return @bitCast(std.mem.nativeToBig(u32, v)); }
fn btn(v: [4]u8) u32 { return @bitCast(std.mem.bigToNative(u32, @bitCast(v))); }
fn m2(val: i8) u8 { return @intCast(val + 2); }
fn m4(val: i8) u8 { return @intCast(val + 8); }
fn m6(val: i8) u8 { return @intCast(val + 32); }
// zig fmt: on
fn checkDiff(diff: irgba) bool {
    const l = diff > @as(irgba, @splat(-0x3));
    const r = diff < @as(irgba, @splat(0x02));
    return @reduce(.And, l & r);
}
fn checkLuma(diff2: irgba) bool {
    const l = diff2 > irgba{ -0x9, -0x21, -0x9, std.math.minInt(i8) };
    const r = diff2 < irgba{ 0x08, 0x020, 0x08, std.math.maxInt(i8) };
    return @reduce(.And, l & r);
}

test "encode decode encode" {
    var rand = std.Random.DefaultPrng.init(0);
    var alloc = std.testing.allocator;

    std.debug.print("--\n", .{});
    for (0..100) |seed| {
        {
            rand.seed(seed);
            const rng = rand.random();

            const width = rng.intRangeAtMost(u32, 100, 1000);
            const height = rng.intRangeAtMost(u32, 100, 1000);
            const channels = rng.enumValue(Channels);
            const num_pixels = @intFromEnum(channels) * @as(usize, width) * @as(usize, height);
            var image = QOI{
                .width = width,
                .height = height,
                .channels = channels,
                .colorspace = rng.enumValue(ColorSpace),
                .pixel_data = try alloc.alloc(u8, num_pixels),
            };
            std.debug.print("image.width = {}\n", .{image.width});
            std.debug.print("image.height = {}\n", .{image.height});
            std.debug.print("image.channels = {}\n", .{image.channels});
            std.debug.print("image.colorspace = {}\n", .{image.colorspace});
            std.debug.print("image.pixels.len = {}\n\n", .{num_pixels});
            defer image.deinit(alloc);

            var encoded = try encode(image, alloc);
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

    for (0..10) |seed| {
        {
            rand.seed(seed);
            const rng = rand.random();

            var qoi: QOI = undefined;

            qoi.width = rng.intRangeAtMost(u32, 0, std.math.maxInt(u12));
            qoi.height = rng.intRangeAtMost(u32, 0, std.math.maxInt(u12));
            qoi.channels = rng.enumValue(Channels);
            qoi.colorspace = rng.enumValue(ColorSpace);
            qoi.pixel_data = try alloc.alloc(u8, @intFromEnum(qoi.channels) * @as(usize, qoi.width * qoi.height));
            defer qoi.deinit(alloc);

            var encoded = try qoi.encode(alloc);
            defer encoded.deinit(alloc);

            var image = try decode(alloc, encoded.items, .{});
            defer image.deinit(alloc);
        }
    }
}
