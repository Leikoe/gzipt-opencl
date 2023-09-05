#define WINDOW_SIZE 256
#define LOOKAHEAD_SIZE 16

// Structure to represent an LZ77 token
struct Token {
    int distance;  // Offset to the matching string in the window
    int length;    // Length of the matching string
    char nextChar; // Next character after the matching string
};

// Function to find the length of a C-style string (null-terminated)
int stringLength(const char* str) {
    int length = 0;
    while (str[length] != '\0') {
        length++;
    }
    return length;
}

// Function to compare two strings
int stringCompare(const char* str1, const char* str2, int length) {
    for (int i = 0; i < length; ++i) {
        if (str1[i] != str2[i]) {
            return 0; // Strings are different
        }
    }
    return 1; // Strings are the same
}

// Function to copy a string
void stringCopy(char* dest, const char* src, int length) {
    for (int i = 0; i < length; ++i) {
        dest[i] = src[i];
    }
    dest[length] = '\0'; // Null-terminate the copied string
}

// Function to find the longest matching string in the window
struct Token findLongestMatch(const char* window, const char* lookahead) {
    struct Token token = {0, 0, lookahead[0]};
    int maxMatchLength = 0;

    for (int i = 0; i < WINDOW_SIZE && i < stringLength(lookahead); ++i) {
        for (int j = 0; j < i && i + j < stringLength(lookahead); ++j) {
            if (lookahead[j] != window[i + j]) {
                break;
            }
            if (j + 1 > maxMatchLength) {
                maxMatchLength = j + 1;
                token.distance = WINDOW_SIZE - i + j;
                token.length = maxMatchLength;
                token.nextChar = lookahead[maxMatchLength];
            }
        }
    }

    return token;
}

// Simple memcpy implementation
void memcpy(void* dest, global const void* src, size_t n) {
    char* destPtr = (char*)dest;
    global const char* srcPtr = (global const char*)src;

    for (size_t i = 0; i < n; i++) {
        destPtr[i] = srcPtr[i];
    }
}

// This is a simplified OpenCL kernel for LZ77 compression.
__kernel void nomnom_(
        global char const* strings,
        global int  const* lens,
        global int  const* offsets,
        global int* compressed_lens
) {
    int gid = get_global_id(0); // Get the global ID of the work item.

    char buff[3000] = {0};
    char input[3000] = {0}; // Input string to compress
    memcpy(&input, strings+offsets[gid], lens[gid]);
    input[lens[gid]] = '\0';

    char window[WINDOW_SIZE] = {0};   // Sliding window
    char lookahead[LOOKAHEAD_SIZE] = {0}; // Look-ahead buffer
    struct Token token;

    int inputLength = stringLength(input);

    for (int i = 0; i < inputLength;) {
        // Fill the look-ahead buffer
        stringCopy(lookahead, input + i, LOOKAHEAD_SIZE);

        // Find the longest matching string in the window
        token = findLongestMatch(window, lookahead);

        // Output the token (distance, length, nextChar)
//        printf("<%d,%d,%c>", token.distance, token.length, token.nextChar);

        // Slide the window and move the input pointer
        int slideDistance = token.length + 1;
        for (int j = 0; j < WINDOW_SIZE - slideDistance; ++j) {
            window[j] = window[j + slideDistance];
        }
        stringCopy(window + WINDOW_SIZE - slideDistance, lookahead, slideDistance);
        i += slideDistance;
    }

//    printf("\n");
}

typedef unsigned char uint8_t;
typedef signed short int16_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned int uint_t;
typedef long int64_t;
typedef unsigned long uint64_t;
typedef int bool_t;

uint32_t lz77_compress (uint8_t *uncompressed_text, uint32_t uncompressed_size, uint8_t *compressed_text, uint8_t pointer_length_width) {
    uint16_t pointer_pos, temp_pointer_pos, output_pointer, pointer_length, temp_pointer_length;
    uint32_t compressed_pointer, output_size, coding_pos, output_lookahead_ref, look_behind, look_ahead;
    uint16_t pointer_pos_max, pointer_length_max;
    pointer_pos_max = (uint16_t)pow((float)2, (float)16 - pointer_length_width);
    pointer_length_max = (uint16_t)pow((float)2, (float)pointer_length_width);

    *((uint32_t *) compressed_text) = uncompressed_size;
    *(compressed_text + 4) = pointer_length_width;
    compressed_pointer = output_size = 5;

    for (coding_pos = 0; coding_pos < uncompressed_size; ++coding_pos) {
        pointer_pos = 0;
        pointer_length = 0;
        for (temp_pointer_pos = 1;
             (temp_pointer_pos < pointer_pos_max) && (temp_pointer_pos <= coding_pos); ++temp_pointer_pos) {
            look_behind = coding_pos - temp_pointer_pos;
            look_ahead = coding_pos;
            for (temp_pointer_length = 0;
                 uncompressed_text[look_ahead++] == uncompressed_text[look_behind++]; ++temp_pointer_length)
                if (temp_pointer_length == pointer_length_max)
                    break;
            if (temp_pointer_length > pointer_length) {
                pointer_pos = temp_pointer_pos;
                pointer_length = temp_pointer_length;
                if (pointer_length == pointer_length_max)
                    break;
            }
        }
        coding_pos += pointer_length;
        if ((coding_pos == uncompressed_size) && pointer_length) {
            output_pointer = (pointer_length == 1) ? 0 : ((pointer_pos << pointer_length_width) | (pointer_length - 2));
            output_lookahead_ref = coding_pos - 1;
        } else {
            output_pointer = (pointer_pos << pointer_length_width) | (pointer_length ? (pointer_length - 1) : 0);
            output_lookahead_ref = coding_pos;
        }
        *((uint16_t * )(compressed_text + compressed_pointer)) = output_pointer;
        compressed_pointer += 2;
        *(compressed_text + compressed_pointer++) = *(uncompressed_text + output_lookahead_ref);
        output_size += 3;
    }

    return output_size;
}

__kernel void nomnom(
        global char const* strings,
        global int  const* lens,
        global int  const* offsets,
        global int* compressed_lens
) {
    int gid = get_global_id(0); // Get the global ID of the work item.
    unsigned char buff[3000] = {0};
    unsigned char input[3000] = {0}; // Input string to compress
    memcpy(&input, strings+offsets[gid], lens[gid]);

    compressed_lens[gid] = (int)lz77_compress(input, lens[gid], buff, 5);
}