import duckdb

col = duckdb.ColumnExpression
lit = duckdb.ConstantExpression
when = duckdb.CaseExpression
all = duckdb.StarExpression
func = duckdb.FunctionExpression
fn_once = duckdb.LambdaExpression
coalesce = duckdb.CoalesceOperator
raw = duckdb.SQLExpression
from_arrow = duckdb.from_arrow
from_query = duckdb.from_query
Relation = duckdb.DuckDBPyRelation

type SqlExpr = duckdb.Expression
