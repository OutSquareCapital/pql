"""Data types Mapping for PQL."""

from duckdb import sqltypes

DataType = sqltypes.DuckDBPyType
Boolean = sqltypes.BOOLEAN
Float32 = sqltypes.FLOAT
Float64 = sqltypes.DOUBLE
Int8 = sqltypes.TINYINT
Int16 = sqltypes.SMALLINT
Int32 = sqltypes.INTEGER
Int64 = sqltypes.BIGINT
UInt8 = sqltypes.UTINYINT
UInt16 = sqltypes.USMALLINT
UInt32 = sqltypes.UINTEGER
UInt64 = sqltypes.UBIGINT

String = sqltypes.VARCHAR
Date = sqltypes.DATE
