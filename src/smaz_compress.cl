#define NULL 0

int memcmp_constant(__constant void *s1, const void *s2, int len) {
    __constant unsigned char *p = (__constant unsigned char *) s1;
    unsigned char *q = (unsigned char *) s2;
    int charCompareStatus = 0;
    //If both pointer pointing same memory block
    while (len > 0) {
        if (*p != *q) {
            //compare the mismatching character
            charCompareStatus = (*p > *q) ? 1 : -1;
            break;
        }
        len--;
        p++;
        q++;
    }
    return charCompareStatus;
}

void memcpy_global(void* dest, global const void* src, size_t n) {
    char* destPtr = (char*)dest;
    global const char* srcPtr = (global const char*)src;

    for (size_t i = 0; i < n; i++) {
        destPtr[i] = srcPtr[i];
    }
}

void memcpy(void* dest, const void* src, size_t n) {
    char* destPtr = (char*)dest;
    const char* srcPtr = (const char*)src;

    for (size_t i = 0; i < n; i++) {
        destPtr[i] = srcPtr[i];
    }
}

void *memset(void *s, int c,  unsigned int len)
{
    unsigned char* p=s;
    while(len--)
    {
        *p++ = (unsigned char)c;
    }
    return s;
}

int memcmp(const void *s1, const void *s2, int len) {
    unsigned char *p = (unsigned char *) s1;
    unsigned char *q = (unsigned char *) s2;
    int charCompareStatus = 0;
    //If both pointer pointing same memory block
    if (s1 == s2) {
        return charCompareStatus;
    }
    while (len > 0) {
        if (*p != *q) {
            //compare the mismatching character
            charCompareStatus = (*p > *q) ? 1 : -1;
            break;
        }
        len--;
        p++;
        q++;
    }
    return charCompareStatus;
}

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

#define STRING0 "\002s,\266",
#define STRING1 "\003had\232\002leW",
#define STRING2 "\003on \216",
#define STRING3 "",
#define STRING4 "\001yS",
#define STRING5 "\002ma\255\002li\227",
#define STRING6 "\003or \260",
#define STRING7 "",
#define STRING8 "\002ll\230\003s t\277",
#define STRING9 "\004fromg\002mel",
#define STRING10 "", "\003its\332",
#define STRING11 "\001z\333",
#define STRING12 "\003ingF",
#define STRING13 "\001>\336",
#define STRING14 "\001 \000\003   (\002nc\344",
#define STRING15 "\002nd=\003 on\312",
#define STRING16 "\002ne\213\003hat\276\003re q",
#define STRING17 "",
#define STRING18 "\002ngT\003herz\004have\306\003s o\225",
#define STRING19 "",
#define STRING20 "\003ionk\003s a\254\002ly\352",
#define STRING21 "\003hisL\003 inN\003 be\252",
#define STRING22 "",
#define STRING23 "\003 fo\325\003 of \003 ha\311",
#define STRING24 "",
#define STRING25 "\002of\005",
#define STRING26 "\003 co\241\002no\267\003 ma\370",
#define STRING27 "",
#define STRING28 "",
#define STRING29 "\003 cl\356\003enta\003 an7",
#define STRING30 "\002ns\300\001\"e",
#define STRING31 "\003n t\217\002ntP\003s, \205",
#define STRING32 "\002pe\320\003 we\351\002om\223",
#define STRING33 "\002on\037",
#define STRING34 "",
#define STRING35 "\002y G",
#define STRING36 "\003 wa\271",
#define STRING37 "\003 re\321\002or*",
#define STRING38 "",
#define STRING39 "\002=\"\251\002ot\337",
#define STRING40 "\003forD\002ou[",
#define STRING41 "\003 toR",
#define STRING42 "\003 th\r",
#define STRING43 "\003 it\366",
#define STRING44 "\003but\261\002ra\202\003 wi\363\002</\361",
#define STRING45 "\003 wh\237",
#define STRING46 "\002  4",
#define STRING47 "\003nd ?",
#define STRING48 "\002re!",
#define STRING49 "",
#define STRING50 "\003ng c",
//#define STRING "",
//#define STRING "\003ly \307\003ass\323\001a\004\002rir",
//#define STRING "",
//#define STRING "",
//#define STRING "",
//#define STRING "\002se_",
//#define STRING "\003of \"",
//#define STRING "\003div\364\002ros\003ere\240",
//#define STRING "",
//#define STRING "\002ta\310\001bZ\002si\324",
//#define STRING "",
//#define STRING "\003and\a\002rs\335",
//#define STRING "\002rt\362",
//#define STRING "\002teE",
//#define STRING "\003ati\316",
//#define STRING "\002so\263",
//#define STRING "\002th\021",
//#define STRING "\002tiJ\001c\034\003allp",
//#define STRING "\003ate\345",
//#define STRING "\002ss\246",
//#define STRING "\002stM",
//#define STRING "",
//#define STRING "\002><\346",
//#define STRING "\002to\024",
//#define STRING "\003arew",
//#define STRING "\001d\030",
//#define STRING "\002tr\303",
//#define STRING "",
//#define STRING "\001\n1\003 a \222",
//#define STRING "\003f tv\002veo",
//#define STRING "\002un\340",
//#define STRING "",
//#define STRING "\003e o\242",
//#define STRING "\002a \243\002wa\326\001e\002",
//#define STRING "\002ur\226\003e a\274",
//#define STRING "\002us\244\003\n\r\n\247",
//#define STRING "\002ut\304\003e c\373",
//#define STRING "\002we\221",
//#define STRING "",
//#define STRING "",
//#define STRING "\002wh\302",
//#define STRING "\001f,",
//#define STRING "",
//#define STRING "",
//#define STRING "",
//#define STRING "\003d t\206",
//#define STRING "",
//#define STRING "",
//#define STRING "\003th \343",
//#define STRING "\001g;",
//#define STRING "",
//#define STRING "",
//#define STRING "\001\r9\003e s\265",
//#define STRING "\003e t\234",
//#define STRING "",
//#define STRING "\003to Y",
//#define STRING "\003e\r\n\236",
//#define STRING "\002d \036\001h\022",
//#define STRING "",
//#define STRING "\001,Q",
//#define STRING "\002 a\031",
//#define STRING "\002 b^",
//#define STRING "\002\r\n\025\002 cI",
//#define STRING "\002 d\245",
//#define STRING "\002 e\253",
//#define STRING "\002 fh\001i\b\002e \v",
//#define STRING "",
//#define STRING "\002 hU\001-\314",
//#define STRING "\002 i8",
//#define STRING "",
//#define STRING "",
//#define STRING "\002 l\315",
//#define STRING "\002 m{",
//#define STRING "\002f :\002 n\354",
//#define STRING "\002 o\035",
//#define STRING "\002 p}\001.n\003\r\n\r\250",
//#define STRING "",
//#define STRING "\002 r\275",
//#define STRING "\002 s>",
//#define STRING "\002 t\016",
//#define STRING "",
//#define STRING "\002g \235\005which+\003whi\367",
//#define STRING "\002 w5",
//#define STRING "\001/\305",
//#define STRING "\003as \214",
//#define STRING "\003at \207",
//#define STRING "",
//#define STRING "\003who\331",
//#define STRING "",
//#define STRING "\001l\026\002h \212",
//#define STRING "",
//#define STRING "\002, $",
//#define STRING "",
//#define STRING "\004withV",
//#define STRING "",
//#define STRING "",
//#define STRING "",
//#define STRING "\001m-",
//#define STRING "",
//#define STRING "",
//#define STRING "\002ac\357",
//#define STRING "\002ad\350",
//#define STRING "\003TheH",
//#define STRING "",
//#define STRING "",
//#define STRING "\004this\233\001n\t",
//#define STRING "",
//#define STRING "\002. y",
//#define STRING "",
//#define STRING "\002alX\003e, \365",
//#define STRING "\003tio\215\002be\\",
//#define STRING "\002an\032\003ver\347",
//#define STRING "",
//#define STRING "\004that0\003tha\313\001o\006",
//#define STRING "\003was2",
//#define STRING "\002arO",
//#define STRING "\002as.",
//#define STRING "\002at'\003the\001\004they\200\005there\322\005theird",
//#define STRING "\002ce\210",
//#define STRING "\004were]",
//#define STRING "",
//#define STRING "\002ch\231\002l \264\001p<",
//#define STRING "",
//#define STRING "",
//#define STRING "\003one\256",
//#define STRING "",
//#define STRING "\003he \023\002dej",
//#define STRING "\003ter\270",
//#define STRING "\002cou",
//#define STRING "",
//#define STRING "\002by\177\002di\201\002eax",
//#define STRING "",
//#define STRING "\002ec\327",
//#define STRING "\002edB",
//#define STRING "\002ee\353",
//#define STRING "",
//#define STRING "",
//#define STRING "\001r\f\002n )",
//#define STRING "",
//#define STRING "",
//#define STRING "",
//#define STRING "\002el\262",
//#define STRING "",
//#define STRING "\003in i\002en3",
//#define STRING "",
//#define STRING "\002o `\001s\n",
//#define STRING "",
//#define STRING "\002er\033",
//#define STRING "\003is t\002es6",
//#define STRING "",
//#define STRING "\002ge\371",
//#define STRING "\004.com\375",
//#define STRING "\002fo\334\003our\330",
//#define STRING "\003ch \301\001t\003",
//#define STRING "\002hab",
//#define STRING "",
//#define STRING "\003men\374",
//#define STRING "",
//#define STRING "\002he\020",
//#define STRING "",
//#define STRING "",
//#define STRING "\001u&",
//#define STRING "\002hif",
//#define STRING "",
//#define STRING "\003not\204\002ic\203",
//#define STRING "\003ed @\002id\355",
//#define STRING "",
//#define STRING "",
//#define STRING "\002ho\273",
//#define STRING "\002r K\001vm",
//#define STRING "",
//#define STRING "",
//#define STRING "",
//#define STRING "\003t t\257\002il\360",
//#define STRING "\002im\342",
//#define STRING "\003en \317\002in\017",
//#define STRING "\002io\220",
//#define STRING "\002s \027\001wA",
//#define STRING "",
//#define STRING "\003er |",
//#define STRING "\003es ~\002is%",
//#define STRING "\002it/",
//#define STRING "",
//#define STRING "\002iv\272",
//#define STRING "",
//#define STRING "\002t #\ahttp://C\001x\372",
//#define STRING "\002la\211",
//#define STRING "\001<\341",
//#define STRING "\003, a\224"

///* Our compression codebook, used for compression */
//__constant char *Smaz_cb[241] = {
//
//};

/* Our compression codebook, used for compression */
__constant char *Smaz_cb[] = {
        "\002s,\266", "\003had\232\002leW", "\003on \216", "", "\001yS",
        "\002ma\255\002li\227", "\003or \260", "", "\002ll\230\003s t\277",
        "\004fromg\002mel", "", "\003its\332", "\001z\333", "\003ingF", "\001>\336",
        "\001 \000\003   (\002nc\344", "\002nd=\003 on\312",
        "\002ne\213\003hat\276\003re q", "", "\002ngT\003herz\004have\306\003s o\225",
        "", "\003ionk\003s a\254\002ly\352", "\003hisL\003 inN\003 be\252", "",
        "\003 fo\325\003 of \003 ha\311", "", "\002of\005",
        "\003 co\241\002no\267\003 ma\370", "", "", "\003 cl\356\003enta\003 an7",
        "\002ns\300\001\"e", "\003n t\217\002ntP\003s, \205",
        "\002pe\320\003 we\351\002om\223", "\002on\037", "", "\002y G", "\003 wa\271",
        "\003 re\321\002or*", "", "\002=\"\251\002ot\337", "\003forD\002ou[",
        "\003 toR", "\003 th\r", "\003 it\366",
        "\003but\261\002ra\202\003 wi\363\002</\361", "\003 wh\237", "\002  4",
        "\003nd ?", "\002re!", "", "\003ng c", "",
        "\003ly \307\003ass\323\001a\004\002rir", "", "", "", "\002se_", "\003of \"",
        "\003div\364\002ros\003ere\240", "", "\002ta\310\001bZ\002si\324", "",
        "\003and\a\002rs\335", "\002rt\362", "\002teE", "\003ati\316", "\002so\263",
        "\002th\021", "\002tiJ\001c\034\003allp", "\003ate\345", "\002ss\246",
        "\002stM", "", "\002><\346", "\002to\024", "\003arew", "\001d\030",
        "\002tr\303", "", "\001\n1\003 a \222", "\003f tv\002veo", "\002un\340", "",
        "\003e o\242", "\002a \243\002wa\326\001e\002", "\002ur\226\003e a\274",
        "\002us\244\003\n\r\n\247", "\002ut\304\003e c\373", "\002we\221", "", "",
        "\002wh\302", "\001f,", "", "", "", "\003d t\206", "", "", "\003th \343",
        "\001g;", "", "", "\001\r9\003e s\265", "\003e t\234", "", "\003to Y",
        "\003e\r\n\236", "\002d \036\001h\022", "", "\001,Q", "\002 a\031", "\002 b^",
        "\002\r\n\025\002 cI", "\002 d\245", "\002 e\253", "\002 fh\001i\b\002e \v",
        "", "\002 hU\001-\314", "\002 i8", "", "", "\002 l\315", "\002 m{",
        "\002f :\002 n\354", "\002 o\035", "\002 p}\001.n\003\r\n\r\250", "",
        "\002 r\275", "\002 s>", "\002 t\016", "", "\002g \235\005which+\003whi\367",
        "\002 w5", "\001/\305", "\003as \214", "\003at \207", "", "\003who\331", "",
        "\001l\026\002h \212", "", "\002, $", "", "\004withV", "", "", "", "\001m-", "",
        "", "\002ac\357", "\002ad\350", "\003TheH", "", "", "\004this\233\001n\t",
        "", "\002. y", "", "\002alX\003e, \365", "\003tio\215\002be\\",
        "\002an\032\003ver\347", "", "\004that0\003tha\313\001o\006", "\003was2",
        "\002arO", "\002as.", "\002at'\003the\001\004they\200\005there\322\005theird",
        "\002ce\210", "\004were]", "", "\002ch\231\002l \264\001p<", "", "",
        "\003one\256", "", "\003he \023\002dej", "\003ter\270", "\002cou", "",
        "\002by\177\002di\201\002eax", "", "\002ec\327", "\002edB", "\002ee\353", "",
        "", "\001r\f\002n )", "", "", "", "\002el\262", "", "\003in i\002en3", "",
        "\002o `\001s\n", "", "\002er\033", "\003is t\002es6", "", "\002ge\371",
        "\004.com\375", "\002fo\334\003our\330", "\003ch \301\001t\003", "\002hab", "",
        "\003men\374", "", "\002he\020", "", "", "\001u&", "\002hif", "",
        "\003not\204\002ic\203", "\003ed @\002id\355", "", "", "\002ho\273",
        "\002r K\001vm", "", "", "", "\003t t\257\002il\360", "\002im\342",
        "\003en \317\002in\017", "\002io\220", "\002s \027\001wA", "", "\003er |",
        "\003es ~\002is%", "\002it/", "", "\002iv\272", "",
        "\002t #\ahttp://C\001x\372", "\002la\211", "\001<\341", "\003, a\224"
};

int smaz_compress(char *in, int inlen, char *out, int outlen) {
    unsigned int h1,h2,h3=0;
    int verblen = 0, _outlen = outlen;
    char verb[256], *_out = out;

    while(inlen) {
        int j = 7, needed;
        char *flush = NULL;
        __constant char *slot;

        h1 = h2 = in[0]<<3;
        if (inlen > 1) h2 += in[1];
        if (inlen > 2) h3 = h2^in[2];
        if (j > inlen) j = inlen;

        /* Try to lookup substrings into the hash table, starting from the
         * longer to the shorter substrings */
        for (; j > 0; j--) {
            switch(j) {
                case 1: slot = Smaz_cb[h1%241]; break;
                case 2: slot = Smaz_cb[h2%241]; break;
                default: slot = Smaz_cb[h3%241]; break;
            }
            while(slot[0]) {
                if (slot[0] == j && memcmp_constant(slot+1,in,j) == 0) {
                    /* Match found in the hash table,
                     * prepare a verbatim bytes flush if needed */
                    if (verblen) {
                        needed = (verblen == 1) ? 2 : 2+verblen;
                        flush = out;
                        out += needed;
                        outlen -= needed;
                    }
                    /* Emit the byte */
                    if (outlen <= 0) return _outlen+1;
                    out[0] = slot[slot[0]+1];
                    out++;
                    outlen--;
                    inlen -= j;
                    in += j;
                    goto out;
                } else {
                    slot += slot[0]+2;
                }
            }
        }
        /* Match not found - add the byte to the verbatim buffer */
        verb[verblen] = in[0];
        verblen++;
        inlen--;
        in++;
        out:
        /* Prepare a flush if we reached the flush length limit, and there
         * is not already a pending flush operation. */
        if (!flush && (verblen == 256 || (verblen > 0 && inlen == 0))) {
            needed = (verblen == 1) ? 2 : 2+verblen;
            flush = out;
            out += needed;
            outlen -= needed;
            if (outlen < 0) return _outlen+1;
        }
        /* Perform a verbatim flush if needed */
        if (flush) {
            if (verblen == 1) {
                flush[0] = (signed char)254;
                flush[1] = verb[0];
            } else {
                flush[0] = (signed char)255;
                flush[1] = (signed char)(verblen-1);
                memcpy(flush+2,verb,verblen);
            }
            flush = NULL;
            verblen = 0;
        }
    }
    return out-_out;
}

#define MAX_BUFF_SIZE 8000

__kernel void compress_kernel(
        __global char const* strings,
        __global int  const* lens,
        __global int  const* offsets,
        __global int* compressed_lens
) {
    int gid = get_global_id(0); // Get the global ID of the work item.
    char input[MAX_BUFF_SIZE] = {0}; // Input string to compress
    char output[MAX_BUFF_SIZE] = {0}; // Output buffer
    memcpy_global(&input, strings+offsets[gid], lens[gid]);

    compressed_lens[gid] = smaz_compress(output, lens[gid], input, MAX_BUFF_SIZE);
}