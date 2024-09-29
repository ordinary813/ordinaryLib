#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

// BMP file header structure
#pragma pack(push, 1)
struct BMPHeader {
    uint16_t signature;       // BM
    uint32_t fileSize;        // Total file size
    uint16_t reserved1;       // Reserved (unused)
    uint16_t reserved2;       // Reserved (unused)
    uint32_t dataOffset;      // Offset to image data
    uint32_t headerSize;      // Size of the BMP header
    uint32_t width;           // Image width
    uint32_t height;          // Image height
    uint16_t planes;          // Number of color planes (must be 1)
    uint16_t bitDepth;        // Bits per pixel
    uint32_t compression;     // Compression method
    uint32_t dataSize;        // Size of raw image data
    uint32_t hResolution;     // Horizontal resolution (pixels per meter)
    uint32_t vResolution;     // Vertical resolution (pixels per meter)
    uint32_t colors;          // Number of colors in the palette
    uint32_t importantColors; // Number of important colors
};
#pragma pack(pop)

int main() {
    std::ifstream file("golden.bmp", std::ios::binary);

    if (!file) {
        std::cerr << "Error: Couldn't open the BMP file." << std::endl;
        return -1;
    }

    // Read the BMP header
    BMPHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (header.signature != 0x4D42) { // Check if it's a valid BMP file (BM)
        std::cerr << "Error: Not a valid BMP file." << std::endl;
        return -1;
    }

    // Check if the BMP file is uncompressed
    if (header.compression != 0) {
        std::cerr << "Error: Compressed BMP files are not supported." << std::endl;
        return -1;
    }

    // Check if the BMP file is 24-bit
    if (header.bitDepth != 24) {
        std::cerr << "Error: Only 24-bit BMP files are supported." << std::endl;
        return -1;
    }

    // Calculate padding
    int padding = (4 - (header.width * 3) % 4) % 4;

    // Read pixel data
    std::vector<uint8_t> pixelData(header.width * header.height * 3); // RGB
    file.seekg(header.dataOffset, std::ios::beg);
    file.read(reinterpret_cast<char*>(pixelData.data()), pixelData.size());

    // Close the file
    file.close();

    // At this point, you have access to the pixel data in the vector 'pixelData'
    // You can manipulate or display it as needed

    return 0;
}
