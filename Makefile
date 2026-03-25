all: debug

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
SCENE_DIR = crown
SCENE_NAME = crown.pbrt
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
SCENE = $(PBRT_SCENES_DIR)/$(strip $(SCENE_DIR))/$(SCENE_NAME)
IMAGE =  $(SCENE_NAME:.pbrt=.exr)
IMAGE_PBRT = $(IMAGE)

#SCENE = ~/src/moana/island/pbrt-v4/island.pbrt
#IMAGE = gonzales.exr
#IMAGE_PBRT = pbrt.exr

PFM = $(IMAGE:.exr=.pfm)

OPTIONS = $(SINGLERAY) $(SYNC) $(VERBOSE) $(QUICK) $(PARSE) $(WRITE_GONZALES) $(USE_GONZALES)

.PHONY: all c ca clean clean_all e edit es editScene em editMakefile lint lldb p perf tags t test \
	test_debug test_release v view wc

PBRT_OPTIONS = #--quiet # --stats #--gpu #--nthreads 1 #--quiet --v 2

OS = $(shell uname)
HOSTNAME = $(shell hostname)

ifeq ($(OS), Darwin)
	SWIFT			= /usr/bin/swift
	VIEWER			= open
	PBRT			= ../../src/pbrt-v3/build/Release/pbrt
	DERIVED_DATA		= ~/Library/Developer/Xcode/DerivedData
	DESTINATION_DIRECTORY   = $(lastword $(shell ls -ltrh $(DERIVED_DATA)|tail -n1))
	BUILD_DIRECTORY = $(DERIVED_DATA)/$(DESTINATION_DIRECTORY)/Build
	RELEASE_DIRECTORY = $(BUILD_DIRECTORY)/Products/Release
	DEBUG_DIRECTORY = $(BUILD_DIRECTORY)/Products/Debug
	GONZALES_RELEASE = $(RELEASE_DIRECTORY)/gonzales
	GONZALES_DEBUG = $(DEBUG_DIRECTORY)/gonzales
	BUILD_DEBUG = xcodebuild -configuration Debug -scheme gonzales -destination 'platform=OS X,arch=x86_64' build
	BUILD_RELEASE = xcodebuild -configuration Release -scheme gonzales -destination 'platform=OS X,arch=x86_64' build
else
	VIEWER 			= loupe
	PBRT 			= ~/src/pbrt-v4/gonsolo/pbrt
	LLDB 			= /usr/lib/swift/bin/lldb
	#LLDB 			= lldb
	ifeq ($(HOSTNAME), Limone)
		SWIFT		= ~/bin/swift
	else
		SWIFT		= swift
endif
	#SWIFT_VERBOSE		= -v
	#SWIFT_EXPORT_DYNAMIC	= -Xlinker --export-dynamic # For stack traces
	#SWIFT_NO_WHOLE_MODULE	= -Xswiftc -no-whole-module-optimization
	SWIFT_LTO 		= --experimental-lto-mode full
	#SWIFT_DEBUG_INFO	= -Xswiftc -g
	#OSSA 			= -Xswiftc -Xfrontend -Xswiftc -enable-ossa-modules
	#SWIFT_ANNOTATIONS 	= -Xswiftc -experimental-performance-annotations
	SWIFT_OPTIMIZE		= $(SWIFT_NO_WHOLE_MODULE) $(SWIFT_DEBUG_INFO) $(OSSA) $(SWIFT_LTO)

	# Should not be needed since there is only one module
	# CROSS 			= -Xswiftc -cross-module-optimization

	#CXX_INTEROP 		= -Xswiftc -cxx-interoperability-mode=default
	#EXPERIMENTAL 		= -Xswiftc -enable-experimental-feature -Xswiftc ExistentialAny

	#SWIFT_CONCURRENCY 	= -Xswiftc -swift-version -Xswiftc 6


	#SWIFT_SUPPRESS 		= -Xswiftc -suppress-warnings
	#WARNINGS_AS_ERRORS 	= -Xswiftc -warnings-as-errors
	DEBUG_OPTIONS   	= $(UPCOMING_FEATURE) $(SWIFT_CONCURRENCY) $(SWIFT_SUPPRESS) $(SWIFT_VERBOSE) \
				  $(SWIFT_EXPORT_DYNAMIC) $(SWIFT_ANNOTATIONS) $(CXX_INTEROP) $(EXPERIMENTAL) \
				  $(WARNINGS_AS_ERRORS)
	RELEASE_OPTIONS 	= $(DEBUG_OPTIONS) $(SWIFT_OPTIMIZE)
	BUILD			= $(SWIFT) build
	BUILD_DEBUG		= $(BUILD) -c debug $(DEBUG_OPTIONS)
	BUILD_RELEASE		= $(BUILD) -c release $(RELEASE_OPTIONS)

	BUILD_DIRECTORY 	= .build
	RELEASE_DIRECTORY 	= $(BUILD_DIRECTORY)/release
	DEBUG_DIRECTORY 	= $(BUILD_DIRECTORY)/debug
	GONZALES_RELEASE 	= $(RELEASE_DIRECTORY)/gonzales
	GONZALES_DEBUG		= $(DEBUG_DIRECTORY)/gonzales
endif
RUN_DEBUG	= @ $(GONZALES_DEBUG) $(OPTIONS) $(SCENE)
RUN_RELEASE	= @ $(GONZALES_RELEASE) $(OPTIONS) $(SCENE)

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

r: release
release:
	@$(BUILD_RELEASE)
d: debug
debug:
	@$(BUILD_DEBUG)
t: test
td: test_debug
test_debug: debug
	@$(RUN_DEBUG)
tr: test_release
test_release: release
	@$(RUN_RELEASE)
tags:
	ctags -R Sources
	


c: clean
clean:
	@rm -rf .build/debug .build/release
	@rm -f cornell-box.png cornell-box.exr cornell-box.hpm cornell-box.tiff tags

ca: clean_all
clean_all:
	@$(SWIFT) package clean
	@rm -f cornell-box.png cornell-box.exr cornell-box.hpm cornell-box.tiff tags
	@rm -rf gonzales.xcodeproj flame.svg perf.data perf.data.old Package.resolved
	@rm -f $(EMBEDDED_C) .build/devicePrograms.ptx

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

vr: view_release
view_release: release
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
	fi; \
	read -p "Run with Gonzales or PBRT? (g/p) [g]: " engine; \
	if [ "$$engine" = "p" ] || [ "$$engine" = "P" ]; then \
		$(PBRT) $(PBRT_OPTIONS) "$$SCENE"; \
	else \
		$(GONZALES_RELEASE) $(OPTIONS) "$$SCENE"; \
	fi; \
	$(VIEWER) "$$IMAGE"

vd: debug
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
	fi; \
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

xcode:
	$(SWIFT) package generate-xcodeproj --xcconfig-overrides Config.xcconfig

FILES=$(shell find Sources -name \*.swift -o -name \*.h -o -name \*.cc| grep -Ev \.build | wc -l)
LINES=$(shell wc -l $$(find Sources -name \*.swift -not -name SobolMatrices.swift -o -name \*.h -o -name \*.cc) | tail -n1 | awk '{ print $$1 }')
wc:
	@echo $(FILES) "files"
	@echo $(LINES) "lines"

# To be able to use perf the following has to be done:
# sudo sysctl -w kernel.perf_event_paranoid=0
# sudo sh -c " echo 0 > /proc/sys/kernel/kptr_restrict"
PERF_RECORD_OPTIONS = -g --freq=99 --call-graph dwarf
PERF_REPORT_OPTIONS = --no-children --percent-limit 1
p: perf
perf: release
	sudo perf record $(PERF_RECORD_OPTIONS) -- $(GONZALES_RELEASE) $(OPTIONS) $(SCENE)
	sudo chown gonsolo perf.data
	perf report $(PERF_REPORT_OPTIONS)
pr: perf_report
perf_report:
	perf report $(PERF_REPORT_OPTIONS)

# Check for memory leaks
leak:
	#valgrind --suppressions=valgrind.supp --gen-suppressions=yes --leak-check=full .build/release/gonzales $(OPTIONS) $(SCENE)
	valgrind --gen-suppressions=yes --leak-check=full .build/release/gonzales $(OPTIONS) $(SCENE)

# Check memory usage
MASSIF_OUT=massif.out.gonzales
memcheck: release
	echo $(GONZALES_RELEASE) $(OPTIONS) $(SCENE)
	valgrind --massif-out-file=$(MASSIF_OUT) --tool=massif $(GONZALES_RELEASE) $(OPTIONS) $(SCENE)
	massif-visualizer $(MASSIF_OUT)

# Record memory while rendering with:
# while true; do ps aux|grep gonzales|grep -Ev grep|awk '{print $5}' >> gonzales_memory; sleep 5; done

flame:
	perf script | stackcollapse-perf.pl | swift demangle | flamegraph.pl --width 10000 --height 48 > flame.svg
	eog -f flame.svg

format:
	@clang-format -i $(shell find Sources -name \*.h -o -name \*.cc)
	@swift-format -i -p $(shell find Sources Tests -name \*.swift -not -name SobolMatrices.swift)
lint:
	swiftlint Sources
codespell:
	codespell -L inout Sources
lldb:
	#$(LLDB) .build/release/gonzales -- $(SINGLERAY) $(SCENE)
	$(LLDB) .build/debug/gonzales -- $(SINGLERAY) $(SCENE)

heaptrack:
	heaptrack $(GONZALES_RELEASE) $(SCENE)

gonzales.perfscript: perf.data
	perf script > gonzales.perfscript
open_trace_in_ui:
	curl -OL https://github.com/google/perfetto/raw/main/tools/open_trace_in_ui
perfetto: gonzales.perfscript open_trace_in_ui
	python open_trace_in_ui -i $<

coated: debug
	.build/debug/testCoated
lldb_coated: debug
	LD_LIBRARY_PATH=. $(LLDB) .build/debug/testCoated

testsuite:
	swift test

book:
	python3 docs/build_book.py

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

