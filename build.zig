const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const libqoi = b.addLibrary(.{
        .name = "qoi",
        .linkage = .static,
        .root_module = b.addModule("qoi", .{
            .root_source_file = b.path("qoi.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    b.installArtifact(libqoi);

    const mod_tests = b.addTest(.{ .root_module = libqoi.root_module });
    const run_mod_tests = b.addRunArtifact(mod_tests);
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
}
