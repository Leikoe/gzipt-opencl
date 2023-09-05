// This is a simplified OpenCL kernel for LZ77 compression.
__kernel void nomnom(
    global char const* strings,
    global int  const* lens,
    global int  const* offsets,
    global int* compressed_lens
) {
    int gid = get_global_id(0); // Get the global ID of the work item.

    char buff[3000] = {0};

    int outputIndex = 0;
    int inputIndex = offsets[gid];
    int inputLength = lens[gid];

    while (inputIndex < inputLength) {
        int maxLength = min(258, inputLength - inputIndex + 1);
        int bestLength = 0;
        int bestOffset = 0;

        for (int j = 1; j < maxLength; ++j) {
            int k = 0;
            while (k < inputIndex && strings[inputIndex - k - 1] == strings[inputIndex + j - k - 1]) {
                k++;
            }
            if (k >= bestLength) {
                bestLength = k;
                bestOffset = j;
            }
        }

        if (bestLength >= 3) {
            buff[outputIndex++] = (char)((bestOffset - 1) & 0xFF);
            buff[outputIndex++] = (char)((((bestOffset - 1) >> 8) & 0x07) | ((bestLength - 3) << 3));
            inputIndex += bestLength;
        } else {
            buff[outputIndex++] = strings[inputIndex++];
        }
    }

    // Update the output information.
    compressed_lens[gid] = outputIndex; // Compressed size
}