all: test_release

#SINGLERAY = --single 32 58
#SYNC = --sync
#VERBOSE = --verbose
#QUICK = --quick
#PARSE = --parse
PTEXMEM = --ptexmem 1 # GB

# 32 of 32 Bitterli scenes from benedikt-bitterli.me/resources rendered successfully:
# bathroom living-room bedroom kitchen staircase2 staircase bathroom2 living-room-2 living-room-3
# dining-room glass-of-water car2 car coffee lamp hair-curl curly-hair straight-hair house spaceship
# classroom dragon teapot-full teapot cornell-box volumetric-caustic water-caustic veach-ajar
# veach-bidir veach-mis material-testball furball
#BITTERLI = ~/src/bitterli
#SCENE_NAME = cornell-box
#SCENE_NAME = layered-cornell-box
#SCENE_NAME = bathroom
#SCENE = $(BITTERLI)/$(SCENE_NAME)/pbrt/scene-v4.pbrt
#SCENE = Scenes/$(SCENE_NAME).pbrt
#IMAGE =  $(SCENE_NAME).exr
#IMAGE_PBRT = $(IMAGE)

# Render 27/27 scenes
PBRT_SCENES_DIR = Scenes/pbrt-v4-scenes
#SCENE_DIR = barcelona-pavilion 		 # 1/27
#SCENE_NAME = pavilion-day.pbrt
#SCENE_DIR = bistro
#SCENE_NAME = bistro_boulangerie.pbrt
#SCENE_DIR = dambreak
#SCENE_NAME = dambreak0.pbrt
#SCENE_NAME = dambreak1.pbrt
#SCENE_DIR = bmw-m6
#SCENE_DIR = bunny-cloud
#SCENE_DIR = bunny-fur
#SCENE_DIR = clouds
#SCENE_DIR = contemporary-bathroom
#SCENE_DIR = crown
#SCENE_NAME = crown.pbrt
#SCENE_DIR = disney-cloud 			10/27
#IMAGE = disney-cloud-720p.exr
#SCENE_DIR = explosion
#SCENE_DIR = ganesha
#SCENE_DIR = hair
#SCENE_NAME  = hair-actual-bsdf.pbrt
#SCENE_DIR = head
#SCENE_DIR = killeroos
#SCENE_NAME  = killeroo-simple.pbrt
#SCENE_DIR = kroken
#SCENE_NAME  = camera-1.pbrt
#SCENE_DIR = landscape
#SCENE_NAME = view-0.pbrt
#SCENE_DIR = lte-orb
#SCENE_NAME = lte-orb-silver.pbrt
#SCENE_DIR = pbrt-book
#SCENE_NAME = book.pbrt
#SCENE_DIR = sanmiguel 				20/27
#SCENE_NAME = sanmiguel-courtyard-second.pbrt
#SCENE_DIR = smoke-plume
#SCENE_NAME = plume.pbrt
#SCENE_DIR = sportscar
#SCENE_NAME = sportscar-area-lights.pbrt
#SCENE_DIR = sssdragon
#SCENE_NAME = dragon_10.pbrt
#SCENE_DIR = transparent-machines
#SCENE_NAME = frame1266.pbrt
#SCENE_DIR = villa
#SCENE_NAME = villa-daylight.pbrt
#SCENE_DIR = watercolor
#SCENE_NAME = camera-1.pbrt
#SCENE_DIR = zero-day 				27/27
#SCENE_NAME = frame120.pbrt
#SCENE_NAME ?= $(SCENE_DIR).pbrt
SCENE = Scenes/cornell-box.pbrt
IMAGE = cornell-box.exr
IMAGE_PBRT = $(IMAGE)

#SCENE = ~/src/moana/island/pbrt-v4/island.pbrt
#IMAGE = gonzales.exr
#IMAGE_PBRT = pbrt.exr

PFM = $(IMAGE:.exr=.pfm)

OPTIONS = $(SINGLERAY) $(SYNC) $(VERBOSE) $(QUICK) $(PARSE) $(WRITE_GONZALES) $(USE_GONZALES)

.PHONY: all c ca clean clean_all e edit es editScene em editMakefile lldb p perf tags t test \
	test_debug test_release v view wc book book-html book-watch ut unittest \
	sms_mitsuba_ref

PBRT_OPTIONS = #--quiet # --stats #--gpu #--nthreads 1 #--quiet --v 2

VIEWER			= loupe
PBRT			= ~/src/pbrt-v4/gonsolo/pbrt
LLDB			= lldb

BUILD_DIR		= build
GONZALES		= $(BUILD_DIR)/gonzales
GONZALES_RELEASE	= $(GONZALES)
GONZALES_DEBUG		= $(GONZALES)

RUN_DEBUG	= @ $(GONZALES) $(OPTIONS) $(SCENE)
RUN_RELEASE	= @ $(GONZALES) $(OPTIONS) $(SCENE)

test: test_debug
v: view
view: view_debug

e: edit
edit:
	@vim
es: editScene
editScene:
	@vim $(SCENE)
em: editMakefile
editMakefile:
	@vim Makefile

OIIO_BRIDGE_SRC = src/oiio/oiio.cc
OIIO_BRIDGE_INC = src/oiio
OIIO_BRIDGE_LIB = $(BUILD_DIR)/liboiiobridge.so

$(OIIO_BRIDGE_LIB): $(OIIO_BRIDGE_SRC) $(OIIO_BRIDGE_INC)/oiio.h
	@mkdir -p $(BUILD_DIR)
	g++ -fPIC -shared -std=c++20 -I$(OIIO_BRIDGE_INC) $(OIIO_BRIDGE_SRC) -lOpenImageIO -o $(OIIO_BRIDGE_LIB)

VIEWER_SRC = src/viewer/viewer.cpp
VIEWER_INC = src/viewer
VIEWER_GEN = src/viewer/generated
VIEWER_LIB = $(BUILD_DIR)/libvulkanviewer.so

$(VIEWER_LIB): $(VIEWER_SRC) $(VIEWER_INC)/viewer.h
	@mkdir -p $(BUILD_DIR)
	g++ -fPIC -shared -std=c++20 -I$(VIEWER_INC) -I$(VIEWER_GEN) \
		$(VIEWER_SRC) -lvulkan -lglfw -o $(VIEWER_LIB)

# Task #162: headless Vulkan ray-query backend (2nd GPU intersection
# backend, alongside the existing software-BVH CUDA path) -- separate
# bridge from VIEWER_LIB since it needs its own headless VkInstance/
# VkDevice (no GLFW/swapchain dependency; most gonzales renders run
# without a window at all). See project_vulkan_rt_backend memory.
VULKANRT_SRC = src/vulkanrt/vulkanrt.cpp
VULKANRT_INC = src/vulkanrt
VULKANRT_GEN = src/vulkanrt/generated
VULKANRT_LIB = $(BUILD_DIR)/libvulkanrt.so

$(VULKANRT_LIB): $(VULKANRT_SRC) $(VULKANRT_INC)/vulkanrt.h $(VULKANRT_GEN)/smoke_comp_spv.h $(VULKANRT_GEN)/trace_ray_comp_spv.h $(VULKANRT_GEN)/intersect_batch_comp_spv.h
	@mkdir -p $(BUILD_DIR)
	g++ -fPIC -shared -std=c++20 -I$(VULKANRT_INC) -I$(VULKANRT_GEN) \
		$(VULKANRT_SRC) -lvulkan -o $(VULKANRT_LIB)

# Task #163 stage 1: CUDA/Vulkan GPU-side interop foundation (real CUDA
# external-memory/semaphore import, no CPU round trip) -- the prerequisite
# for a Vulkan RT backend that's actually faster than gonzales's existing
# software-BVH GPU path, not just a correctness proof (--vulkan-rt-shade's
# naive host-round-trip approach is ~94x slower). See
# project_vulkan_rt_backend memory.
CUDA_HOME ?= /opt/cuda
VULKANINTEROP_SRC = src/vulkaninterop/vulkaninterop.cpp
VULKANINTEROP_INC = src/vulkaninterop
VULKANINTEROP_GEN = src/vulkaninterop/generated
VULKANINTEROP_LIB = $(BUILD_DIR)/libvulkaninterop.so

# The interop bridge is the one component that hard-requires the CUDA
# toolkit. Without it the whole build used to fail -- the Mojo link line
# always pulls in -lvulkaninterop, so a missing cuda_runtime.h took down the
# CPU renderer too (this is what had CI red: its container has no CUDA).
# Detect the toolkit and fall back to vulkaninterop_stub.cpp, which exports
# the same symbols and reports "unavailable" the way callers already expect
# from a driver lacking the interop extensions.
HAVE_CUDA := $(wildcard $(CUDA_HOME)/include/cuda_runtime.h)

ifeq ($(HAVE_CUDA),)
$(VULKANINTEROP_LIB): $(VULKANINTEROP_INC)/vulkaninterop_stub.cpp $(VULKANINTEROP_INC)/vulkaninterop.h
	@mkdir -p $(BUILD_DIR)
	@echo "note: no CUDA at $(CUDA_HOME) -- building vulkaninterop stub (CUDA/Vulkan interop disabled)"
	g++ -fPIC -shared -std=c++20 -I$(VULKANINTEROP_INC) \
		$(VULKANINTEROP_INC)/vulkaninterop_stub.cpp -o $(VULKANINTEROP_LIB)
else
$(VULKANINTEROP_LIB): $(VULKANINTEROP_SRC) $(VULKANINTEROP_INC)/vulkaninterop.h $(VULKANINTEROP_GEN)/interop_double_comp_spv.h $(VULKANINTEROP_GEN)/intersect_batch_comp_spv.h
	@mkdir -p $(BUILD_DIR)
	g++ -fPIC -shared -std=c++20 -I$(VULKANINTEROP_INC) -I$(VULKANINTEROP_GEN) -I$(CUDA_HOME)/include \
		$(VULKANINTEROP_SRC) -lvulkan -L$(CUDA_HOME)/lib64 -lcudart -o $(VULKANINTEROP_LIB)
endif

ifdef GITHUB_ACTIONS
MOJO_BUILD_FLAGS = --target-accelerator sm_89 --target-cpu x86-64-v3
else
MOJO_BUILD_FLAGS = --target-accelerator sm_86
endif
MOJO_LINK_FLAGS = -Xlinker -L$(BUILD_DIR) -Xlinker -loiiobridge -Xlinker -lvulkanviewer \
                  -Xlinker -lvulkanrt -Xlinker -lvulkaninterop \
                  -Xlinker -rpath -Xlinker $(BUILD_DIR) -Xlinker -lm

MOJO_SRCS := $(wildcard src/gonzales/*.mojo)
$(GONZALES): $(MOJO_SRCS) pyproject.toml $(OIIO_BRIDGE_LIB) $(VIEWER_LIB) $(VULKANRT_LIB) $(VULKANINTEROP_LIB)
	@mkdir -p $(BUILD_DIR)
	uv run mojo build src/gonzales/__init__.mojo -I src -o $(GONZALES) $(MOJO_BUILD_FLAGS) $(MOJO_LINK_FLAGS)

# Standalone clean-room port of the reference renderer's single-scatter SMS
# estimator (Tools/sms_mitsuba_ref.mojo) -- a self-contained cross-check for
# gonzales's own SMS, not part of the renderer. See that file's header.
sms_mitsuba_ref: $(OIIO_BRIDGE_LIB) $(VIEWER_LIB) $(VULKANRT_LIB) $(VULKANINTEROP_LIB) Tools/sms_mitsuba_ref.mojo
	@mkdir -p $(BUILD_DIR)
	uv run mojo build Tools/sms_mitsuba_ref.mojo -I src -o $(BUILD_DIR)/sms_mitsuba_ref $(MOJO_LINK_FLAGS)

r: release
release: $(GONZALES)
d: debug
debug: $(GONZALES)
t: test
td: test_debug
test_debug: debug
	@$(RUN_DEBUG)
tr: test_release
test_release: release
	@$(RUN_RELEASE)

# Unit tests (std.testing.TestSuite) for pure functions — separate from the
# scene-render smoke tests above. Each file under Tests/unit/ is its own
# `mojo run` invocation (TestSuite.discover_tests[__functions_in_module()]).
UNIT_TEST_SRCS := $(filter-out Tests/unit/_%,$(wildcard Tests/unit/*.mojo))

# These three instantiate GPU kernels at COMPILE time (DeviceContext /
# vulkaninterop entry points), so on a machine without the CUDA toolkit they
# fail with "function instantiation failed" before any of their own
# has_accelerator() runtime guards can print "SKIP: no GPU". That is a
# build-environment limit, not a test failure -- the same one that made the
# renderer itself unbuildable without CUDA (see HAVE_CUDA above) -- so drop
# them from the suite when there is no toolkit to build them against. Every
# other GPU-touching test (test_vulkanrt_*, test_gpu_scene_upload) compiles
# fine and skips at runtime, so they stay in.
GPU_ONLY_TEST_SRCS := Tests/unit/test_gpu.mojo \
                      Tests/unit/test_vulkaninterop.mojo \
                      Tests/unit/test_vulkaninterop_rt.mojo
ifeq ($(HAVE_CUDA),)
UNIT_TEST_SRCS := $(filter-out $(GPU_ONLY_TEST_SRCS),$(UNIT_TEST_SRCS))
endif
ut: unittest
# Depends on the OIIO/viewer bridge libs (not the full $(GONZALES) binary)
# because a few tests (e.g. test_gpu_scene_upload.mojo) compile real code
# from pbrt_parser.mojo/shading.mojo that references their C symbols
# (load_texture_rgb, texture, ...) even on a code path that never executes
# them — the JIT still needs the symbols resolvable at link time.
#
# Each file is already an independent `mojo run` process (TestSuite has no
# built-in parallel-execution mode -- checked the actual stdlib source,
# suite.mojo's run() is sequential in-process). Runs all files concurrently,
# bounded by nproc, instead of one at a time: log each to its own file, wait
# for all, then replay the logs back in the original file order so output
# stays readable/deterministic despite running out of order. Fails if any
# file failed (checked after all finish, not fail-fast, so a run tells you
# everything that broke in one pass).
unittest: $(OIIO_BRIDGE_LIB) $(VIEWER_LIB) $(VULKANRT_LIB) $(VULKANINTEROP_LIB)
	@rm -rf build/unittest-logs && mkdir -p build/unittest-logs
	@i=0; \
	for f in $(UNIT_TEST_SRCS); do \
		i=$$((i+1)); \
		echo "$$f" >> build/unittest-logs/order.txt; \
		( uv run mojo run -I src $(MOJO_LINK_FLAGS) "$$f" > build/unittest-logs/$$i.log 2>&1; \
		  echo $$? > build/unittest-logs/$$i.exit ) & \
		while [ $$(jobs -pr | wc -l) -ge $$(nproc) ]; do sleep 0.2; done; \
	done; \
	wait; \
	fail=0; i=0; \
	while IFS= read -r f; do \
		i=$$((i+1)); \
		echo "=== $$f ==="; \
		cat build/unittest-logs/$$i.log; \
		[ "$$(cat build/unittest-logs/$$i.exit)" = "0" ] || fail=1; \
	done < build/unittest-logs/order.txt; \
	exit $$fail

# GPU profiling with Nsight Compute (needs nsight-compute + admin perf-counter
# access). `sudo -E` preserves the env so the Mojo runtime libs load. Profiles
# the hot kernels (traverse/shade) on cornell-box; import the report with:
#   ncu --import build/gonzales.ncu-rep --page details
PROFILE_SCENE ?= Scenes/cornell-box.pbrt
profile: release
	sudo -E ncu --set basic --launch-count 8 -o build/gonzales -f \
		$(GONZALES) --gpu $(PROFILE_SCENE)
# Profile just the shade kernel (registers/occupancy/duration):
profile-shade: release
	sudo -E ncu --set basic --kernel-name "regex:shade_nee" --launch-count 2 \
		-o build/gonzales_shade -f $(GONZALES) --gpu $(PROFILE_SCENE)
	@sudo -E ncu --import build/gonzales_shade.ncu-rep --page details 2>/dev/null \
		| grep -iE "Registers Per Thread|Achieved Occupancy|Duration|Memory Throughput|Compute"

tags:
	ctags -R src
	


c: clean
clean:
	@rm -rf $(BUILD_DIR)
	@rm -f cornell-box.png cornell-box.exr cornell-box.hpm cornell-box.tiff tags

ca: clean_all
clean_all: clean
	@rm -rf flame.svg perf.data perf.data.old

clean-gh-runs:
	gh run list --limit 200 --json databaseId --jq '.[8:] | .[].databaseId' | xargs -I {} gh run delete {}


CONVERT = magick 
DENOISE = oidnDenoise
vn: view_denoised
view_denoised:
	$(CONVERT) -type truecolor -endian LSB $(IMAGE) $(PFM)
	#$(CONVERT) -type truecolor -endian LSB albedo.exr albedo.pfm
	#$(CONVERT) -type truecolor -endian LSB normal.exr normal.pfm
	#$(DENOISE) -hdr $(PFM) -alb albedo.pfm -nrm normal.pfm -o denoised.albnrm.pfm
	$(DENOISE) -hdr $(PFM) -o denoised.pfm
	#gimp denoised.albnrm.pfm
	gimp denoised.pfm

v: view

define SELECT_SCENE
	@read -p "Render Scenes/cornell-box.pbrt? [Y/n] " ans; \
	if [ -z "$$ans" ] || [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
		SCENE="Scenes/cornell-box.pbrt"; \
		IMAGE="cornell-box.exr"; \
	else \
		echo "Select pbrt-v4 scene:"; \
		echo " 1) barcelona-pavilion (default)"; \
		echo " 2) bistro"; \
		echo " 3) contemporary-bathroom"; \
		echo " 4) crown"; \
		echo " 5) hair"; \
		echo " 6) killeroos"; \
		echo " 7) kroken"; \
		echo " 8) landscape"; \
		echo " 9) lte-orb"; \
		echo "10) pbrt-book"; \
		echo "11) sanmiguel"; \
		echo "12) smoke-plume"; \
		echo "13) sportscar"; \
		echo "14) sssdragon"; \
		echo "15) transparent-machines"; \
		echo "16) villa"; \
		echo "17) watercolor"; \
		echo "18) zero-day"; \
		read -p "Enter number [1]: " choice; \
		case "$$choice" in \
			""|1) SCENE="$(PBRT_SCENES_DIR)/barcelona-pavilion/pavilion-day.pbrt"; IMAGE="pavilion-day.exr" ;; \
			2)    SCENE="$(PBRT_SCENES_DIR)/bistro/bistro_boulangerie.pbrt"; IMAGE="bistro_boulangerie.exr" ;; \
			3)    SCENE="$(PBRT_SCENES_DIR)/contemporary-bathroom/contemporary-bathroom.pbrt"; IMAGE="contemporary-bathroom.exr" ;; \
			4)    SCENE="$(PBRT_SCENES_DIR)/crown/crown.pbrt"; IMAGE="crown.exr" ;; \
			5)    SCENE="$(PBRT_SCENES_DIR)/hair/hair-actual-bsdf.pbrt"; IMAGE="hair-actual-bsdf.exr" ;; \
			6)    SCENE="$(PBRT_SCENES_DIR)/killeroos/killeroo-simple.pbrt"; IMAGE="killeroo-simple.exr" ;; \
			7)    SCENE="$(PBRT_SCENES_DIR)/kroken/camera-1.pbrt"; IMAGE="camera-1.exr" ;; \
			8)    SCENE="$(PBRT_SCENES_DIR)/landscape/view-0.pbrt"; IMAGE="view-0.exr" ;; \
			9)    SCENE="$(PBRT_SCENES_DIR)/lte-orb/lte-orb-silver.pbrt"; IMAGE="lte-orb-silver.exr" ;; \
			10)   SCENE="$(PBRT_SCENES_DIR)/pbrt-book/book.pbrt"; IMAGE="book.exr" ;; \
			11)   SCENE="$(PBRT_SCENES_DIR)/sanmiguel/sanmiguel-courtyard-second.pbrt"; IMAGE="sanmiguel-courtyard-second.exr" ;; \
			12)   SCENE="$(PBRT_SCENES_DIR)/smoke-plume/plume.pbrt"; IMAGE="plume.exr" ;; \
			13)   SCENE="$(PBRT_SCENES_DIR)/sportscar/sportscar-area-lights.pbrt"; IMAGE="sportscar-area-lights.exr" ;; \
			14)   SCENE="$(PBRT_SCENES_DIR)/sssdragon/dragon_10.pbrt"; IMAGE="dragon_10.exr" ;; \
			15)   SCENE="$(PBRT_SCENES_DIR)/transparent-machines/frame1266.pbrt"; IMAGE="frame1266.exr" ;; \
			16)   SCENE="$(PBRT_SCENES_DIR)/villa/villa-daylight.pbrt"; IMAGE="villa-daylight.exr" ;; \
			17)   SCENE="$(PBRT_SCENES_DIR)/watercolor/camera-1.pbrt"; IMAGE="camera-1.exr" ;; \
			18)   SCENE="$(PBRT_SCENES_DIR)/zero-day/frame120.pbrt"; IMAGE="frame120.exr" ;; \
			*)    echo "Invalid choice."; exit 1 ;; \
		esac; \
	fi
endef

vr: view_release
view_release: release
	$(SELECT_SCENE); \
	read -p "Run with Gonzales or PBRT? (g/p) [g]: " engine; \
	if [ "$$engine" = "p" ] || [ "$$engine" = "P" ]; then \
		$(PBRT) $(PBRT_OPTIONS) "$$SCENE"; \
	else \
		$(GONZALES_RELEASE) $(OPTIONS) "$$SCENE"; \
	fi; \
	$(VIEWER) "$$IMAGE"

vrg: view_release_gpu
view_release_gpu: release
	$(SELECT_SCENE); \
	read -p "Run with Gonzales or PBRT on GPU? (g/p) [g]: " engine; \
	if [ "$$engine" = "p" ] || [ "$$engine" = "P" ]; then \
		$(PBRT) $(PBRT_OPTIONS) --gpu "$$SCENE"; \
	else \
		$(GONZALES_RELEASE) $(OPTIONS) --gpu "$$SCENE"; \
	fi; \
	$(VIEWER) "$$IMAGE"

vir: view_interactive_release
view_interactive_release: release
	$(SELECT_SCENE); \
	$(GONZALES_RELEASE) --interactive $(OPTIONS) "$$SCENE"

virg: view_interactive_release_gpu
view_interactive_release_gpu: release
	$(SELECT_SCENE); \
	$(GONZALES_RELEASE) --interactive --gpu $(OPTIONS) "$$SCENE"

vd: debug
	$(SELECT_SCENE); \
	read -p "Run with Gonzales or PBRT? (g/p) [g]: " engine; \
	if [ "$$engine" = "p" ] || [ "$$engine" = "P" ]; then \
		$(PBRT) $(PBRT_OPTIONS) "$$SCENE"; \
	else \
		$(GONZALES_DEBUG) $(OPTIONS) "$$SCENE"; \
	fi; \
	$(VIEWER) "$$IMAGE"
vp: view_pbrt
view_pbrt: test_pbrt
	@$(VIEWER) $(IMAGE_PBRT)
tp: test_pbrt
test_pbrt:
	$(PBRT) $(PBRT_OPTIONS) $(SCENE)

FILES=$(shell find src -name \*.mojo -o -name \*.h -o -name \*.cc | wc -l)
LINES=$(shell wc -l $$(find src -name \*.mojo -o -name \*.h -o -name \*.cc) | tail -n1 | awk '{ print $$1 }')
wc:
	@echo "Mojo:  $$(find src -name '*.mojo' | xargs wc -l 2>/dev/null | tail -1 | awk '{print $$1}') lines"
	@echo "C++:   $$(find src -name '*.cc' -o -name '*.cpp' -o -name '*.h' | xargs wc -l 2>/dev/null | tail -1 | awk '{print $$1}') lines"

# To be able to use perf the following has to be done:
# sudo sysctl -w kernel.perf_event_paranoid=0
# sudo sh -c " echo 0 > /proc/sys/kernel/kptr_restrict"
PERF_RECORD_OPTIONS = -g --freq=47 --call-graph dwarf
PERF_REPORT_OPTIONS = --no-children --percent-limit 1
p: perf
perf: release
	$(SELECT_SCENE); \
	sudo perf record $(PERF_RECORD_OPTIONS) -- $(GONZALES_RELEASE) $(OPTIONS) "$$SCENE"; \
	sudo chown gonsolo perf.data; \
	perf report $(PERF_REPORT_OPTIONS)
pr: perf_report
perf_report:
	perf report $(PERF_REPORT_OPTIONS)

# Check for memory leaks
leak:
	valgrind --gen-suppressions=yes --leak-check=full $(GONZALES) $(OPTIONS) $(SCENE)

# Check memory usage
MASSIF_OUT=massif.out.gonzales
memcheck: release
	echo $(GONZALES_RELEASE) $(OPTIONS) $(SCENE)
	valgrind --massif-out-file=$(MASSIF_OUT) --tool=massif $(GONZALES_RELEASE) $(OPTIONS) $(SCENE)
	massif-visualizer $(MASSIF_OUT)

# Record memory while rendering with:
# while true; do ps aux|grep gonzales|grep -Ev grep|awk '{print $5}' >> gonzales_memory; sleep 5; done

flame:
	perf script | stackcollapse-perf.pl | flamegraph.pl --width 10000 --height 48 > flame.svg
	eog -f flame.svg

format:
	@clang-format -i $(shell find src -name \*.h -o -name \*.cc)
codespell:
	codespell -L inout src
lldb:
	$(LLDB) $(GONZALES) -- $(SINGLERAY) $(SCENE)

heaptrack:
	heaptrack $(GONZALES_RELEASE) $(SCENE)

gonzales.perfscript: perf.data
	perf script > gonzales.perfscript
open_trace_in_ui:
	curl -OL https://github.com/google/perfetto/raw/main/tools/open_trace_in_ui
perfetto: gonzales.perfscript open_trace_in_ui
	python open_trace_in_ui -i $<

# ── Book build ──────────────────────────────────────────────────────────────
# Usage:
#   make book          — render Markdown with live code snippets inlined
#   make book-html     — also produce HTML via Pandoc (needs: sudo apt install pandoc)
#   make book-watch    — live-rebuild on source/doc changes (needs: pip install watchdog)

book:
	python3 docs/build_book.py
	@echo "Book written to docs/book/"

book-html:
	python3 docs/build_book.py --html
	@echo "HTML book written to docs/book/*.html"

book-watch:
	python3 docs/build_book.py --watch --html

WALL_SCENE = wall.pbrt
WALL_IMAGE = pavilion-wall.exr
WALL_IMAGE_GONZALES = wall-gonzales.exr
WALL_IMAGE_PBRT = wall-pbrt.exr

tw: test_wall
test_wall: debug
	@ $(GONZALES_DEBUG) $(OPTIONS) $(WALL_SCENE)
	@mv $(WALL_IMAGE) $(WALL_IMAGE_GONZALES)

twp: test_wall_pbrt
test_wall_pbrt:
	$(PBRT) $(PBRT_OPTIONS) $(WALL_SCENE)
	@mv $(WALL_IMAGE) $(WALL_IMAGE_PBRT)

cw: compare_wall
compare_wall: test_wall test_wall_pbrt
	python3 Scripts/compare_exr.py $(WALL_IMAGE_GONZALES) $(WALL_IMAGE_PBRT)

