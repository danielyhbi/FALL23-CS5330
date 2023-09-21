# # OSX compiler
# #CC = clang++

# # Dwarf compiler
# CC = /Applications/Xcode.app/Contents/Developer/usr/bin/g++

# CXX = $(CC)

# # OSX include paths (for MacPorts)
# #CFLAGS = -I/opt/local/include -I../include

# # OSX include paths (for homebrew, probably)
# CFLAGS = -Wc++11-extensions -std=c++11 -I/usr/local/include/opencv4 -I../include -DENABLE_PRECOMPILED_HEADERS=OFF

# # Dwarf include paths
# #CFLAGS = -I../include # opencv includes are in /usr/include
# CXXFLAGS = $(CFLAGS)

# # OSX Library paths (if you use MacPorts)
# #LDFLAGS = -L/opt/local/lib

# #OSX Library paths (if you use homebrew, probably)
# #LDFLAGS = -L/usr/local/lib

# # Dwarf Library paths
# LDFLAGS = -L/usr/local/lib/opencv4/3rdparty -L/usr/local/lib # opencv libraries are here

# # opencv libraries
# LDLIBS = -ltiff -lpng -ljpeg -llapack -lblas -lz -ljasper -lwebp -lIlmImf -lgs -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework CoreServices -framework CoreGraphics -framework AppKit -framework OpenCL  -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect

# main: main.o
# 	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)


# # BINDIR = ../bin

# # vid: vidDisplay.o filters.o
# # 	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

# # face: faceDetect.o filters.o
# # 	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

# # imod: imgMod.o
# # 	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

# # macbeth: macbeth.o
# # 	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

# clean:
# 	rm -f *.o *~ 
