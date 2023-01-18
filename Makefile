all: d

#SINGLERAY = --single 20 20
#SYNC = --sync
VERBOSE = --verbose
#QUICK = --quick
#PARSE = --parse
PTEXMEM = --ptexmem 1 # GB

SCENE = ../../src/bitterli/cornell-box/pbrt/scene-v4.pbrt
IMAGE =  cornell-box.exr
#SCENE = ../../src/bitterli/bathroom/pbrt/scene-v4.pbrt
#IMAGE =  bathroom.exr
#SCENE = ~/src/pbrt-v4-scenes/killeroos/killeroo-coated-gold.pbrt
#IMAGE = killeroo-coated-gold.exr

PFM = $(IMAGE:.exr=.pfm)

OPTIONS = $(SINGLERAY) $(SYNC) $(VERBOSE) $(QUICK) $(PARSE) $(WRITE_GONZALES) $(USE_GONZALES)

.PHONY: all c clean e edit es editScene em editMakefile lldb p perf tags t test \
	test_unchecked test_debug test_release v view wc openexr

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
	ifeq ($(HOSTNAME), Limone)
		SWIFT		= ~/bin/swift
	else
		SWIFT		= swift
	endif
	#SWIFT_VERBOSE		= -v
	SWIFT_EXPORT_DYNAMIC	= -Xlinker --export-dynamic # For stack traces
	#SWIFT_NO_WHOLE_MODULE	= -Xswiftc -no-whole-module-optimization
	#SWIFT_DEBUG_INFO	= -Xswiftc -g
	SWIFT_OPTIMIZE_FLAG	= -Xswiftc -Ounchecked -Xcc -Xclang -Xcc -target-feature -Xcc -Xclang -Xcc +avx2
	#OSSA 			= -Xswiftc -Xfrontend -Xswiftc -enable-ossa-modules
	SWIFT_ANNOTATIONS 	= -Xswiftc -experimental-performance-annotations
	SWIFT_OPTIMIZE		= $(SWIFT_OPTIMIZE_FLAG) $(SWIFT_NO_WHOLE_MODULE) $(SWIFT_DEBUG_INFO) $(OSSA)
	LINK_PTEX		= -Xlinker -L -Xlinker ../../src/ptex/build/src/ptex/ -Xlinker -lPtex
	CXX_INTEROP 		= -Xswiftc -enable-experimental-cxx-interop
	DEBUG_OPTIONS   	= $(SWIFT_VERBOSE) $(SWIFT_EXPORT_DYNAMIC) $(LINK_PTEX) $(SWIFT_ANNOTATIONS) $(CXX_INTEROP)
	RELEASE_OPTIONS 	= $(SWIFT_VERBOSE) $(SWIFT_EXPORT_DYNAMIC) $(LINK_PTEX) $(SWIFT_OPTIMIZE) $(SWIFT_ANNOTATIONS) $(CXX_INTEROP)
	BUILD			= $(SWIFT) build
	BUILD_DEBUG		= $(BUILD) -c debug $(DEBUG_OPTIONS)
	BUILD_RELEASE		= $(BUILD) -c release $(RELEASE_OPTIONS)

	BUILD_DIRECTORY 	= .build
	RELEASE_DIRECTORY 	= $(BUILD_DIRECTORY)/release
	DEBUG_DIRECTORY 	= $(BUILD_DIRECTORY)/debug
	GONZALES_RELEASE 	= $(RELEASE_DIRECTORY)/gonzales
	GONZALES_DEBUG		= $(DEBUG_DIRECTORY)/gonzales
endif
LD_LIB = LD_LIBRARY_PATH=Extern/openexr/build/_deps/imath-build/src/Imath:Extern/openexr/build/src/lib/OpenEXR:Extern/ptex/build/src/ptex
RUN_DEBUG	= @ $(LD_LIB) $(GONZALES_DEBUG) $(OPTIONS) $(SCENE)
RUN_RELEASE	= @ $(LD_LIB) $(GONZALES_RELEASE) $(OPTIONS) $(SCENE)

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
openexr: Extern/openexr/build/src/lib/OpenEXR/libOpenEXR-3_2.so
Extern/openexr/build/src/lib/OpenEXR/libOpenEXR-3_2.so:
	@mkdir -p Extern/openexr/build
	@cd Extern/openexr/build; cmake ..; make -j8 -s
ptex: Extern/ptex/build/src/ptex/libPtex.so
Extern/ptex/build/src/ptex/libPtex.so:
	@mkdir -p Extern/ptex/build
	@cd Extern/ptex/build; cmake ..; make -j8 -s

r: release
release: ptex openexr
	@$(BUILD_RELEASE)
d: debug
debug: ptex openexr
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

#CONVERT = magick convert
CONVERT = convert
DENOISE = oidnDenoise
vn: view_denoised
view_denoised:
	$(CONVERT) -type truecolor -endian LSB $(IMAGE) $(PFM)
	$(CONVERT) -type truecolor -endian LSB albedo.exr albedo.pfm
	$(CONVERT) -type truecolor -endian LSB normal.exr normal.pfm
	$(DENOISE) -hdr $(PFM) -alb albedo.pfm -nrm normal.pfm -o denoised.albnrm.pfm
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
	@$(VIEWER) $(IMAGE)
tp: test_pbrt
test_pbrt:
	$(PBRT) $(PBRT_OPTIONS) $(SCENE)

xcode:
	$(SWIFT) package generate-xcodeproj --xcconfig-overrides Config.xcconfig

FILES=$(shell find Sources -name \*.swift -o -name \*.h -o -name \*.cpp| egrep -v \.build | wc -l)
LINES=$(shell wc -l $$(find Sources -name \*.swift -o -name \*.h -o -name \*.cpp) | tail -n1 | awk '{ print $$1 }')
wc:
	@echo $(FILES) "files"
	@echo $(LINES) "lines"

# To be able to use perf the following has to be done:
# sudo sysctl -w kernel.perf_event_paranoid=0
# sudo sh -c " echo 0 > /proc/sys/kernel/kptr_restrict"
PERF_RECORD_OPTIONS = --freq=100 --call-graph dwarf
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
	valgrind --suppressions=valgrind.supp --gen-suppressions=yes --leak-check=full .build/release/gonzales $(OPTIONS) $(SCENE)

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
	clang-format --dry-run Sources/exr/include/exr.h Sources/exr/exr.cpp
	clang-format --dry-run Sources/ptex/include/ptex.h Sources/ptex/ptex.cpp
	swift-format lint -r Sources/gonzales/

codespell:
	codespell -L inout Sources
lldb:
	#lldb .build/release/gonzales -- $(SINGLERAY) $(SCENE)
	lldb .build/debug/gonzales -- $(SINGLERAY) $(SCENE)

