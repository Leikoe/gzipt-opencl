#define MAX_ALLOC 3000 // in bytes

/* ------------------- Types and macros */
typedef unsigned char mz_uint8;
typedef signed short mz_int16;
typedef unsigned short mz_uint16;
typedef unsigned int mz_uint32;
typedef unsigned int mz_uint;
typedef long mz_int64;
typedef unsigned long mz_uint64;
typedef int mz_bool;

#define MZ_FALSE (0)
#define MZ_TRUE (1)

#define MZ_MACRO_END while (0)

#define MZ_FILE void *

typedef struct mz_dummy_time_t_tag
{
    mz_uint32 m_dummy1;
    mz_uint32 m_dummy2;
} mz_dummy_time_t;
#define MZ_TIME_T mz_dummy_time_t

#define MZ_ASSERT(x) assert(x)

#define MZ_MALLOC(x) NULL
#define MZ_FREE(x) (void)x, ((void)0)
#define MZ_REALLOC(p, x) NULL

#define MZ_MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MZ_MIN(a, b) (((a) < (b)) ? (a) : (b))

/* ------------------- zlib-style API Definitions. */

/* For more compatibility with zlib, miniz.c uses unsigned long for some parameters/struct members. Beware: mz_ulong can be either 32 or 64-bits! */
typedef unsigned long mz_ulong;

#define MZ_ADLER32_INIT (1)
#define MZ_CRC32_INIT (0)

/* Compression strategies. */
enum
{
    MZ_DEFAULT_STRATEGY = 0,
    MZ_FILTERED = 1,
    MZ_HUFFMAN_ONLY = 2,
    MZ_RLE = 3,
    MZ_FIXED = 4
};

/* Method */
#define MZ_DEFLATED 8

/* Compression levels: 0-9 are the standard zlib-style levels, 10 is best possible compression (not zlib compatible, and may be very slow), MZ_DEFAULT_COMPRESSION=MZ_DEFAULT_LEVEL. */
enum
{
    MZ_NO_COMPRESSION = 0,
    MZ_BEST_SPEED = 1,
    MZ_BEST_COMPRESSION = 9,
    MZ_UBER_COMPRESSION = 10,
    MZ_DEFAULT_LEVEL = 6,
    MZ_DEFAULT_COMPRESSION = -1
};

#define MZ_VERSION "11.0.2"
#define MZ_VERNUM 0xB002
#define MZ_VER_MAJOR 11
#define MZ_VER_MINOR 2
#define MZ_VER_REVISION 0
#define MZ_VER_SUBREVISION 0

#ifndef MINIZ_NO_ZLIB_APIS

/* Flush values. For typical usage you only need MZ_NO_FLUSH and MZ_FINISH. The other values are for advanced use (refer to the zlib docs). */
enum
{
    MZ_NO_FLUSH = 0,
    MZ_PARTIAL_FLUSH = 1,
    MZ_SYNC_FLUSH = 2,
    MZ_FULL_FLUSH = 3,
    MZ_FINISH = 4,
    MZ_BLOCK = 5
};

/* Return status codes. MZ_PARAM_ERROR is non-standard. */
enum
{
    MZ_OK = 0,
    MZ_STREAM_END = 1,
    MZ_NEED_DICT = 2,
    MZ_ERRNO = -1,
    MZ_STREAM_ERROR = -2,
    MZ_DATA_ERROR = -3,
    MZ_MEM_ERROR = -4,
    MZ_BUF_ERROR = -5,
    MZ_VERSION_ERROR = -6,
    MZ_PARAM_ERROR = -10000
};

/* Window bits */
#define MZ_DEFAULT_WINDOW_BITS 15

struct mz_internal_state;

/* Compression/decompression stream struct. */
typedef struct mz_stream_s
{
    const unsigned char *next_in; /* pointer to next byte to read */
    unsigned int avail_in;        /* number of bytes available at next_in */
    mz_ulong total_in;            /* total number of bytes consumed so far */

    unsigned char *next_out; /* pointer to next byte to write */
    unsigned int avail_out;  /* number of bytes that can be written to next_out */
    mz_ulong total_out;      /* total number of bytes produced so far */

    struct mz_internal_state *state; /* internal state, allocated by zalloc/zfree */

    int data_type;     /* data_type (unused) */
    mz_ulong adler;    /* adler32 of the source or uncompressed data */
    mz_ulong reserved; /* not used */
} mz_stream;

typedef mz_stream *mz_streamp;

// *********************************************************************************

void *memset(void *str, int c, size_t n) {
    for (int i=0; i<n; i++) {
        *((char*)str+i) = c;
    }

    return str;
}

int mz_deflateInit(mz_streamp pStream, int level)
{
    return mz_deflateInit2(pStream, level, MZ_DEFLATED, MZ_DEFAULT_WINDOW_BITS, 9, MZ_DEFAULT_STRATEGY);
}

int mz_deflateInit2(mz_streamp pStream, int level, int method, int window_bits, int mem_level, int strategy)
{
    tdefl_compressor *pComp;
    mz_uint comp_flags = TDEFL_COMPUTE_ADLER32 | tdefl_create_comp_flags_from_zip_params(level, window_bits, strategy);

    if (!pStream)
        return MZ_STREAM_ERROR;
    if ((method != MZ_DEFLATED) || ((mem_level < 1) || (mem_level > 9)) || ((window_bits != MZ_DEFAULT_WINDOW_BITS) && (-window_bits != MZ_DEFAULT_WINDOW_BITS)))
        return MZ_PARAM_ERROR;

    pStream->data_type = 0;
    pStream->adler = MZ_ADLER32_INIT;
    pStream->msg = NULL;
    pStream->reserved = 0;
    pStream->total_in = 0;
    pStream->total_out = 0;

    pComp = (tdefl_compressor *)pStream->zalloc(pStream->opaque, 1, sizeof(tdefl_compressor));
    if (!pComp)
        return MZ_MEM_ERROR;

    pStream->state = (struct mz_internal_state *)pComp;

    if (tdefl_init(pComp, NULL, NULL, comp_flags) != TDEFL_STATUS_OKAY)
    {
        mz_deflateEnd(pStream);
        return MZ_PARAM_ERROR;
    }

    return MZ_OK;
}

//int mz_deflateReset(mz_streamp pStream)
//{
//    if ((!pStream) || (!pStream->state) || (!pStream->zalloc) || (!pStream->zfree))
//        return MZ_STREAM_ERROR;
//    pStream->total_in = pStream->total_out = 0;
//    tdefl_init((tdefl_compressor *)pStream->state, NULL, NULL, ((tdefl_compressor *)pStream->state)->m_flags);
//    return MZ_OK;
//}
//
//int mz_deflate(mz_streamp pStream, int flush)
//{
//    size_t in_bytes, out_bytes;
//    mz_ulong orig_total_in, orig_total_out;
//    int mz_status = MZ_OK;
//
//    if ((!pStream) || (!pStream->state) || (flush < 0) || (flush > MZ_FINISH) || (!pStream->next_out))
//        return MZ_STREAM_ERROR;
//    if (!pStream->avail_out)
//        return MZ_BUF_ERROR;
//
//    if (flush == MZ_PARTIAL_FLUSH)
//        flush = MZ_SYNC_FLUSH;
//
//    if (((tdefl_compressor *)pStream->state)->m_prev_return_status == TDEFL_STATUS_DONE)
//        return (flush == MZ_FINISH) ? MZ_STREAM_END : MZ_BUF_ERROR;
//
//    orig_total_in = pStream->total_in;
//    orig_total_out = pStream->total_out;
//    for (;;)
//    {
//        tdefl_status defl_status;
//        in_bytes = pStream->avail_in;
//        out_bytes = pStream->avail_out;
//
//        defl_status = tdefl_compress((tdefl_compressor *)pStream->state, pStream->next_in, &in_bytes, pStream->next_out, &out_bytes, (tdefl_flush)flush);
//        pStream->next_in += (mz_uint)in_bytes;
//        pStream->avail_in -= (mz_uint)in_bytes;
//        pStream->total_in += (mz_uint)in_bytes;
//        pStream->adler = tdefl_get_adler32((tdefl_compressor *)pStream->state);
//
//        pStream->next_out += (mz_uint)out_bytes;
//        pStream->avail_out -= (mz_uint)out_bytes;
//        pStream->total_out += (mz_uint)out_bytes;
//
//        if (defl_status < 0)
//        {
//            mz_status = MZ_STREAM_ERROR;
//            break;
//        }
//        else if (defl_status == TDEFL_STATUS_DONE)
//        {
//            mz_status = MZ_STREAM_END;
//            break;
//        }
//        else if (!pStream->avail_out)
//            break;
//        else if ((!pStream->avail_in) && (flush != MZ_FINISH))
//        {
//            if ((flush) || (pStream->total_in != orig_total_in) || (pStream->total_out != orig_total_out))
//                break;
//            return MZ_BUF_ERROR; /* Can't make forward progress without some input.
// */
//        }
//    }
//    return mz_status;
//}
//
//int mz_deflateEnd(mz_streamp pStream)
//{
//    if (!pStream)
//        return MZ_STREAM_ERROR;
//    if (pStream->state)
//    {
//        pStream->zfree(pStream->opaque, pStream->state);
//        pStream->state = NULL;
//    }
//    return MZ_OK;
//}

mz_ulong mz_deflateBound(mz_streamp pStream, mz_ulong source_len)
{
    (void)pStream;
    /* This is really over conservative. (And lame, but it's actually pretty tricky to compute a true upper bound given the way tdefl's blocking works.) */
    return MZ_MAX(128 + (source_len * 110) / 100, 128 + source_len + ((source_len / (31 * 1024)) + 1) * 5);
}

int mz_compress2(unsigned char *pDest, mz_ulong *pDest_len, const unsigned char *pSource, mz_ulong source_len, int level)
{
    int status;
    mz_stream stream;
    memset(&stream, 0, sizeof(stream));

    /* In case mz_ulong is 64-bits (argh I hate longs). */
    if ((mz_uint64)(source_len | *pDest_len) > 0xFFFFFFFFU)
        return MZ_PARAM_ERROR;

    stream.next_in = pSource;
    stream.avail_in = (mz_uint32)source_len;
    stream.next_out = pDest;
    stream.avail_out = (mz_uint32)*pDest_len;

    status = mz_deflateInit(&stream, level);
    if (status != MZ_OK)
        return status;

    status = mz_deflate(&stream, MZ_FINISH);
    if (status != MZ_STREAM_END)
    {
        mz_deflateEnd(&stream);
        return (status == MZ_OK) ? MZ_BUF_ERROR : status;
    }

    *pDest_len = stream.total_out;
    return mz_deflateEnd(&stream);
}

int mz_compress(unsigned char *pDest, mz_ulong *pDest_len, const unsigned char *pSource, mz_ulong source_len)
{
    return mz_compress2(pDest, pDest_len, pSource, source_len, MZ_DEFAULT_COMPRESSION);
}

mz_ulong mz_compressBound(mz_ulong source_len)
{
    return mz_deflateBound(NULL, source_len);
}

kernel void nomnom(
    global char const* strings,
    global int  const* lens,
    global int  const* offsets,
    global int* compressed_lens
) {
    const size_t i = get_global_id(0);
    ulong cmp_len = mz_compressBound(lens[i]);
    char pCmp[MAX_ALLOC] = {0}; // fuck

    int count = 0;
    for (int k=offsets[i]; k<lens[i]; k++) {
        char c = strings[k];
        if (c == 'a') {
            count += 1;
        }
    }

    compressed_lens[i] = cmp_len;
}