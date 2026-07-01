from statebudgetmem.interfaces import MemoryPiece, ViewType
from statebudgetmem.routing import QueryType as RoutingQueryType, RuleBasedRouter
from statebudgetmem.schemas import QueryType as SchemaQueryType


def test_query_type_is_shared_across_schema_and_routing() -> None:
    assert RoutingQueryType is SchemaQueryType
    assert SchemaQueryType("current") is SchemaQueryType.CURRENT


def test_general_query_routes_to_no_personal_memory() -> None:
    router = RuleBasedRouter()
    assert router.route("北京天气怎么样？") is ViewType.NONE


def test_memory_piece_contract_is_importable() -> None:
    piece = MemoryPiece(content="用户喜欢跑步", timestamp=1.0)
    assert piece.content == "用户喜欢跑步"
