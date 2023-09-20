// This just defines the function table.
// Without this there are link errors.
#include "../../External/Optix/7.7.0/include/optix_function_table_definition.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>

void contextLogCallback(unsigned int level, const char *tag, const char *message, void *)
{
      fprintf( stderr, "[%2d][%12s]: %s\n", (int)level, tag, message );
}

#ifdef __cplusplus
}
#endif

