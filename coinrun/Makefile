UNAME := $(shell uname -s)
OBJDIRR=.build-release
OBJDIRD=.build-debug

ifeq ($(UNAME),Linux)
  PKG  =pkg-config
  MOC  =moc
  LIBS =-L/usr/lib64 -lm -lGL -lGLU
  INC  =-I/usr/include
  SO=.so
endif

ifeq ($(UNAME),Darwin)
  ifneq ("$(wildcard /usr/local/opt/qt5/bin/moc"), "")
    $(info Mac, assumed python3, homebrew branch)
    MOC =/usr/local/opt/qt5/bin/moc
    export PKG_CONFIG_PATH=/usr/local/opt/qt5/lib/pkgconfig
  else
    $(info Mac, assumed python3, anaconda branch)
    MOC =moc
  endif
  PKG   =pkg-config
  SO=.dylib
endif

INC     += `$(PKG) --cflags Qt5Widgets`
LIBS    += -lstdc++ `$(PKG) --libs Qt5Widgets`

CC=gcc
LINK=gcc
AR=ar r
AR_OUT=
LINK_OUT= -o
MINUS_O = -o
CFLAGS   = -std=c++11 -Wall -Wno-unused-variable -Wno-unused-function -Wno-deprecated-register -fPIC -g -O3 -march=native $(INC)
CFLAGSD  = -std=c++11 -Wall -Wno-unused-variable -Wno-unused-function -Wno-deprecated-register -fPIC -g -DDEBUG $(INC)

SHARED  = -shared
DEPENDS = -MMD -MF $@.dep

EVERY_BIN=$(OBJDIRR)/coinrun_cpp$(SO) $(OBJDIRD)/coinrun_cpp_d$(SO)

SRC = \
 coinrun.cpp

SRC_R = $(patsubst %.cpp, $(OBJDIRR)/%.o, $(SRC))
SRC_D = $(patsubst %.cpp, $(OBJDIRD)/%.o, $(SRC))

EVERY_OBJ_R = $(SRC_R)
EVERY_OBJ_D = $(SRC_D)
DEP = $(patsubst %.o,%.o.dep, $(EVERY_OBJ_R) $(EVERY_OBJ_D))

all: dirs $(EVERY_BIN)

$(OBJDIRR)/coinrun.o: .generated/coinrun.moc
.generated/coinrun.moc: coinrun.cpp
	$(MOC) -o $@ $<

$(OBJDIRR)/coinrun_cpp$(SO): $(SRC_R)
	$(LINK) $(SHARED) $(LINK_OUT) $@ $^ $(LIBS)

$(OBJDIRD)/coinrun_cpp_d$(SO): $(SRC_D)
	$(LINK) $(SHARED) $(LINK_OUT) $@ $^ $(LIBS)

$(OBJDIRR)/%.o: %.cpp
	$(CC) $(CFLAGS) -c $<  $(MINUS_O)$@ $(DEPENDS)
$(OBJDIRD)/%.o: %.cpp
	$(CC) $(CFLAGSD) -c $<  $(MINUS_O)$@ $(DEPENDS)

.PHONY: depends clean dirs

clean:
	$(RM) $(EVERY_BIN) $(EVERY_OBJ_R) $(EVERY_OBJ_D) .generated/*.moc *.ilk *.pdb $(DEP)
	rm -rf .generated
	rm -rf $(OBJDIRD)
	rm -rf $(OBJDIRR)

depends:
	cat  $(DEP) > Makefile.dep

.generated:
	mkdir -p .generated
$(OBJDIRR):
	mkdir -p $@
$(OBJDIRD):
	mkdir -p $@

dirs: .generated $(OBJDIRR) $(OBJDIRD)

-include Makefile.dep
