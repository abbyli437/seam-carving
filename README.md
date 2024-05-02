Note: I'm deleting the old readme because it doesn't tell you how to compile but this is what works: 

You need to install opencv locally. After opencv is installed then this works in Ubuntu to compile: 
``` g++ $(pkg-config opencv4 --cflags) -std=c++2a seam-carving.cpp $(pkg-config opencv4 --libs) -o seam-carving```
StackOverflow link found here: https://stackoverflow.com/questions/24337932/cannot-get-opencv-to-compile-because-of-undefined-references

When you run ./seam-carving you need to have the photo you're carving inside the same folder as seam-carving.cpp

Gifs: 
Gifs were done using ImageMagick. If you're using Ubuntu/Linux you should follow this post to install: https://askubuntu.com/questions/1398876/magick-h-no-such-file-or-directory-even-after-installing-magick 

After this is run then you should be able to compile via 
``` g++ $(pkg-config opencv4 --cflags) -std=c++2a seam-carving.cpp $(pkg-config opencv4 --libs) -o seam-carving `Magick++-config --cxxflags --cppflags --ldflags --libs` ```