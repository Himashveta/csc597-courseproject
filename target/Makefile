# Build the mock compiler shared library with gcov coverage instrumentation.
#
# Two targets:
#   make            -> libmock_compiler.so (with --coverage)
#   make clean      -> remove build artefacts and .gcda/.gcno files
#
# The shared library is loaded at runtime via ctypes from
# polyfuzz.target_loader.

CC ?= gcc
# UBSan WITHOUT no-sanitize-recover: a runtime error gets reported but
# the program continues, so __gcov_dump still runs at process exit and
# we get coverage for crashing seeds. The Python harness sets
# UBSAN_OPTIONS=halt_on_error=1 only when explicitly hunting bugs.
CFLAGS  := -O0 -g -fPIC -Wall -Wextra --coverage -fprofile-arcs -ftest-coverage \
           -fsanitize=undefined
LDFLAGS := --coverage -shared -fsanitize=undefined

SRCS := mock_compiler.c mc_probe.c
HDRS := mock_compiler.h mc_probe.h
OBJS := mock_compiler.o mc_probe.o
LIB  := libmock_compiler.so

all: $(LIB)

%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB): $(OBJS)
	$(CC) $(LDFLAGS) -o $(LIB) $(OBJS)

clean:
	rm -f *.o $(LIB) *.gcda *.gcno *.gcov

.PHONY: all clean
