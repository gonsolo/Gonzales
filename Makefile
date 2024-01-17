all: d

#SINGLERAY = --single 68 2
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
BITTERLI = ~/src/bitterli
SCENE_NAME = cornell-box
SCENE = $(BITTERLI)/$(SCENE_NAME)/pbrt/scene-v4.pbrt
IMAGE =  $(SCENE_NAME).exr
IMAGE_PBRT = $(IMAGE)

# Render 27/27 scenes
#PBRT_SCENES_DIR = /home/gonsolo/src/pbrt-v4-scenes
#SCENE_DIR = barcelona-pavilion 		 1/27
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
#SCENE_DIR = disney-cloud 			10/27
#IMAGE = disney-cloud-720p.exr
#SCENE_DIR = explosion
#SCENE_DIR = ganesha
#SCENE_DIR = hair
#SCENE_NAME  = hair-actual-bsdf.pbrt
#SCENE_DIR = head
#SCENE_DIR = killeroos
#SCENE_NAME  = killeroo-simple.pbrt
#SCENE_NAME  = $(SCENE_DIR).pbrt
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
#SCENE = $(PBRT_SCENES_DIR)/$(SCENE_DIR)/$(SCENE_NAME)
#IMAGE =  $(SCENE_NAME:.pbrt=.exr)
#IMAGE_PBRT = $(IMAGE)

#SCENE = ~/src/moana/island/pbrt-v4/island.pbrt
#IMAGE = gonzales.exr
#IMAGE_PBRT = pbrt.exr

PFM = $(IMAGE:.exr=.pfm)

OPTIONS = $(SINGLERAY) $(SYNC) $(VERBOSE) $(QUICK) $(PARSE) $(WRITE_GONZALES) $(USE_GONZALES)

.PHONY: all c clean e edit es editScene em editMakefile lldb p perf tags t test \
	test_unchecked test_debug test_release v view wc

PBRT_OPTIONS = --stats #--gpu #--nthreads 1 #--quiet --v 2

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
	VIEWER 			= gimp
	PBRT 			= ~/bin/pbrt
	LLDB 			= /usr/libexec/swift/bin/lldb
	ifeq ($(HOSTNAME), Limone)
		SWIFT		= ~/bin/swift
	else
		SWIFT		= swift
endif
	#SWIFT_VERBOSE		= -v
	SWIFT_EXPORT_DYNAMIC	= -Xlinker --export-dynamic # For stack traces
	#SWIFT_NO_WHOLE_MODULE	= -Xswiftc -no-whole-module-optimization
	#SWIFT_DEBUG_INFO	= -Xswiftc -g
	SWIFT_OPTIMIZE_FLAG	= #-Xswiftc -Ounchecked # -Xcc -Xclang -Xcc -target-feature -Xcc -Xclang -Xcc +avx2
	#OSSA 			= -Xswiftc -Xfrontend -Xswiftc -enable-ossa-modules
	SWIFT_ANNOTATIONS 	= -Xswiftc -experimental-performance-annotations
	SWIFT_OPTIMIZE		= $(SWIFT_OPTIMIZE_FLAG) $(SWIFT_NO_WHOLE_MODULE) $(SWIFT_DEBUG_INFO) $(OSSA)

	# Should not be needed since there is only one module
	# CROSS 			= -Xswiftc -cross-module-optimization

	#CXX_INTEROP 		= -Xswiftc -enable-experimental-cxx-interop
	CXX_INTEROP 		= -Xswiftc -cxx-interoperability-mode=default
	EXPERIMENTAL 		= -Xswiftc -enable-experimental-feature -Xswiftc ExistentialAny
	DEBUG_OPTIONS   	= $(SWIFT_VERBOSE) $(SWIFT_EXPORT_DYNAMIC) $(SWIFT_ANNOTATIONS) $(CXX_INTEROP) $(EXPERIMENTAL)
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
	@vi
es: editScene
editScene:
	@vi $(SCENE)
em: editMakefile
editMakefile:
	@vi Makefile

NVCC = nvcc
EMBEDDED_C = Sources/cudaBridge/embedded.c
DEVICE_PROGRAMS_SOURCE = Sources/cudaBridge/devicePrograms.cu
DEVICE_PROGRAMS_PTX = .build/devicePrograms.ptx

optix: $(EMBEDDED_C)

$(DEVICE_PROGRAMS_PTX): $(DEVICE_PROGRAMS_SOURCE) Sources/cudaBridge/LaunchParams.h
	@$(NVCC) -allow-unsupported-compiler --ptx -ISources/cudaBridge/include -IExternal/Optix/7.7.0/include/ -rdc=true -o $@ $< > /dev/null

$(EMBEDDED_C): $(DEVICE_PROGRAMS_PTX)
	@bin2c -c --padd 0 --type char --name embedded_ptx_code $(DEVICE_PROGRAMS_PTX) > $(EMBEDDED_C)

r: release
release: optix
	@$(BUILD_RELEASE)
d: debug
debug: optix
	@$(BUILD_DEBUG)
t: test
td: test_debug
test_debug: debug
	@$(RUN_DEBUG)
tr: test_release
test_release: release
	@$(RUN_RELEASE)
tu: test_unchecked
test_unchecked:
	@$(SWIFT) run -c release  $(SWIFT_OPTIONS)-Xswiftc -Ounchecked gonzales $(SCENE)
tags:
	ctags -R Sources
c: clean
clean:
	@$(SWIFT) package clean
	@rm -f cornell-box.png cornell-box.exr cornell-box.hpm cornell-box.tiff tags
	@rm -rf gonzales.xcodeproj flame.svg perf.data perf.data.old Package.resolved
	@rm -f $(EMBEDDED_C) .build/devicePrograms.ptx

#CONVERT = magick convert
CONVERT = convert
DENOISE = oidnDenoise
vn: view_denoised
view_denoised:
	$(CONVERT) -type truecolor -endian LSB $(IMAGE) $(PFM)
	$(CONVERT) -type truecolor -endian LSB albedo.exr albedo.pfm
	$(CONVERT) -type truecolor -endian LSB normal.exr normal.pfm
	#$(DENOISE) -hdr $(PFM) -alb albedo.pfm -nrm normal.pfm -o denoised.albnrm.pfm
	$(DENOISE) -hdr $(PFM) -o denoised.albnrm.pfm
	$(VIEWER) denoised.albnrm.pfm

v: view
vr: view_release
view_release: test_release
	@$(VIEWER) $(IMAGE)
vd: view_debug
view_debug: test_debug
	@$(VIEWER) $(IMAGE)
vp: view_pbrt
view_pbrt: test_pbrt
	@$(VIEWER) $(IMAGE_PBRT)
tp: test_pbrt
test_pbrt:
	$(PBRT) $(PBRT_OPTIONS) $(SCENE)

xcode:
	$(SWIFT) package generate-xcodeproj --xcconfig-overrides Config.xcconfig

FILES=$(shell find Sources -name \*.swift -o -name \*.h -o -name \*.cc| egrep -v \.build | wc -l)
LINES=$(shell wc -l $$(find Sources -name \*.swift -o -name \*.h -o -name \*.cc) | tail -n1 | awk '{ print $$1 }')
wc:
	@echo $(FILES) "files"
	@echo $(LINES) "lines"

# To be able to use perf the following has to be done:
# sudo sysctl -w kernel.perf_event_paranoid=0
# sudo sh -c " echo 0 > /proc/sys/kernel/kptr_restrict"
PERF_RECORD_OPTIONS = --freq=20 --call-graph dwarf
PERF_REPORT_OPTIONS = --no-children --percent-limit 1
p: perf
perf: release
	perf record $(PERF_RECORD_OPTIONS) $(GONZALES_RELEASE) $(OPTIONS) $(SCENE)
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
# while true; do ps aux|grep gonzales|egrep -v grep|awk '{print $5}' >> gonzales_memory; sleep 5; done

flame:
	perf script|  ../../src/FlameGraph/stackcollapse-perf.pl | swift-demangle | ../../src/FlameGraph/flamegraph.pl --width 10000 --height 48 > flame.svg
	eog -f flame.svg

f: format
format:
	@clang-format --dry-run $(shell find Sources -name \*.h -o -name \*.cc)
	@swift-format lint -r Sources/gonzales/

codespell:
	codespell -L inout Sources
lldb:
	$(LLDB) .build/release/gonzales -- $(SINGLERAY) $(SCENE)
	#$(LLDB) .build/debug/gonzales -- $(SINGLERAY) $(SCENE)

