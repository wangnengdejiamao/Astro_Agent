"""
查询 Milvus 中所有 collection 的名称和数据集情况，以及手动清除数据集的功能。
"""

from __future__ import annotations

from typing import Dict, List, Optional

try:
    from pymilvus import Collection, connections, utility

    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    print("警告: pymilvus 未安装，无法使用此脚本")

from .milvus import CollectionType, COLLECTION_CONFIGS, MilvusCollectionManager
from .registry import collection_manager


def list_all_collections() -> List[str]:
    """
    查询 Milvus 中所有 collection 的名称。

    Returns:
        List[str]: 所有 collection 的名称列表
    """
    if not MILVUS_AVAILABLE:
        raise RuntimeError("Milvus 不可用，请安装 pymilvus")

    collection_manager.ensure_connection()
    collections = utility.list_collections()
    return collections


def get_collection_info(collection_name: str) -> Dict:
    """
    获取指定 collection 的详细信息。

    Args:
        collection_name: Collection 名称

    Returns:
        Dict: Collection 信息，包括实体数量、是否加载等
    """
    if not MILVUS_AVAILABLE:
        raise RuntimeError("Milvus 不可用，请安装 pymilvus")

    collection_manager.ensure_connection()
    if not utility.has_collection(collection_name):
        return {"exists": False, "message": f"Collection '{collection_name}' 不存在"}

    collection = Collection(name=collection_name)
    collection.load()
    info = {
        "exists": True,
        "name": collection_name,
        "num_entities": collection.num_entities,
        "is_empty": collection.is_empty,
        "description": collection.description,
    }
    return info


def get_dataset_info(dataset_name: str) -> Dict[str, Dict]:
    """
    查询指定数据集在所有 collection 中的情况。

    Args:
        dataset_name: 数据集名称

    Returns:
        Dict[str, Dict]: 每个 collection 类型对应的数据集信息
    """
    if not MILVUS_AVAILABLE:
        raise RuntimeError("Milvus 不可用，请安装 pymilvus")

    collection_manager.ensure_connection()
    dataset_info = {}

    for collection_type in CollectionType:
        collection_name = collection_manager.get_collection_name(collection_type)
        if not utility.has_collection(collection_name):
            dataset_info[collection_type.value] = {
                "collection_name": collection_name,
                "exists": False,
                "count": 0,
                "message": f"Collection '{collection_name}' 不存在",
            }
            continue

        collection = Collection(name=collection_name)
        collection.load()

        # 查询该数据集在 collection 中的记录数
        try:
            # 先检查是否有数据
            results = collection.query(
                expr=f'dataset_name == "{dataset_name}"',
                output_fields=["id"],
                limit=1,
            )

            if not results:
                dataset_info[collection_type.value] = {
                    "collection_name": collection_name,
                    "exists": True,
                    "count": 0,
                    "count_str": "0",
                    "has_data": False,
                }
                continue

            # 获取总数：查询所有记录（对于大数据集可能较慢）
            # Milvus 的 query limit 最大值为 16384
            max_query_limit = 16384
            all_results = collection.query(
                expr=f'dataset_name == "{dataset_name}"',
                output_fields=["id"],
                limit=max_query_limit,
            )
            count = len(all_results)

            if count >= max_query_limit:
                count_str = f">= {count:,} (可能更多，已限制查询)"
            else:
                count_str = f"{count:,}"

            dataset_info[collection_type.value] = {
                "collection_name": collection_name,
                "exists": True,
                "count": count,
                "count_str": count_str,
                "has_data": count > 0,
            }
        except Exception as e:
            dataset_info[collection_type.value] = {
                "collection_name": collection_name,
                "exists": True,
                "count": 0,
                "error": str(e),
            }

    return dataset_info


def delete_dataset_from_collection(
    collection_type: CollectionType, dataset_name: str, confirm: bool = False
) -> bool:
    """
    从指定的 collection 中删除指定数据集的所有数据。

    Args:
        collection_type: Collection 类型
        dataset_name: 数据集名称
        confirm: 是否确认删除（默认 False，需要显式确认）

    Returns:
        bool: 是否删除成功
    """
    if not MILVUS_AVAILABLE:
        raise RuntimeError("Milvus 不可用，请安装 pymilvus")

    if not confirm:
        raise ValueError("删除操作需要显式确认，请设置 confirm=True")

    collection_manager.ensure_connection()
    collection_name = collection_manager.get_collection_name(collection_type)

    if not utility.has_collection(collection_name):
        print(f"Collection '{collection_name}' 不存在，无需删除")
        return False

    collection = Collection(name=collection_name)
    collection.load()

    # 先查询确认有数据
    existing = collection.query(
        expr=f'dataset_name == "{dataset_name}"',
        output_fields=["id"],
        limit=1,
    )

    if not existing:
        print(f"数据集 '{dataset_name}' 在 collection '{collection_name}' 中不存在")
        return False

    # 删除数据
    try:
        collection.delete(expr=f'dataset_name == "{dataset_name}"')
        collection.flush()
        print(f"成功从 collection '{collection_name}' 中删除数据集 '{dataset_name}'")
        return True
    except Exception as e:
        print(f"删除失败: {e}")
        return False


def delete_dataset_from_all_collections(dataset_name: str, confirm: bool = False) -> Dict[str, bool]:
    """
    从所有 collection 中删除指定数据集的所有数据。

    Args:
        dataset_name: 数据集名称
        confirm: 是否确认删除（默认 False，需要显式确认）

    Returns:
        Dict[str, bool]: 每个 collection 类型的删除结果
    """
    if not confirm:
        raise ValueError("删除操作需要显式确认，请设置 confirm=True")

    results = {}
    for collection_type in CollectionType:
        try:
            success = delete_dataset_from_collection(collection_type, dataset_name, confirm=True)
            results[collection_type.value] = success
        except Exception as e:
            print(f"从 {collection_type.value} 删除失败: {e}")
            results[collection_type.value] = False

    return results


def print_all_collections():
    """打印所有 collection 的名称和基本信息。"""
    print("=" * 60)
    print("Milvus 中的所有 Collection:")
    print("=" * 60)

    collections = list_all_collections()
    if not collections:
        print("  没有找到任何 collection")
        return

    for i, name in enumerate(collections, 1):
        info = get_collection_info(name)
        if info.get("exists"):
            print(f"{i}. {name}")
            print(f"   实体数量: {info.get('num_entities', 'N/A')}")
            print(f"   是否为空: {info.get('is_empty', 'N/A')}")
        else:
            print(f"{i}. {name} (不存在)")
        print()


def print_dataset_info(dataset_name: str):
    """打印指定数据集在所有 collection 中的情况。"""
    print("=" * 60)
    print(f"数据集 '{dataset_name}' 在所有 Collection 中的情况:")
    print("=" * 60)

    info = get_dataset_info(dataset_name)
    for collection_type, data in info.items():
        print(f"\nCollection 类型: {collection_type}")
        print(f"  Collection 名称: {data.get('collection_name', 'N/A')}")
        if data.get("exists"):
            if "error" in data:
                print(f"  错误: {data['error']}")
            else:
                count_str = data.get("count_str", str(data.get("count", 0)))
                print(f"  记录数: {count_str}")
                print(f"  有数据: {data.get('has_data', False)}")
        else:
            print(f"  状态: {data.get('message', 'N/A')}")


def compare_datasets(dataset_name1: str, dataset_name2: str) -> Dict[str, Dict]:
    """
    对比两个数据集在项目使用的 4 个 collection 中的情况。

    Args:
        dataset_name1: 第一个数据集名称
        dataset_name2: 第二个数据集名称

    Returns:
        Dict[str, Dict]: 每个 collection 类型的对比信息
    """
    if not MILVUS_AVAILABLE:
        raise RuntimeError("Milvus 不可用，请安装 pymilvus")

    collection_manager.ensure_connection()
    comparison = {}

    for collection_type in CollectionType:
        collection_name = collection_manager.get_collection_name(collection_type)
        
        if not utility.has_collection(collection_name):
            comparison[collection_type.value] = {
                "collection_name": collection_name,
                "exists": False,
                "dataset1": {"count": 0, "count_str": "0", "has_data": False},
                "dataset2": {"count": 0, "count_str": "0", "has_data": False},
                "message": f"Collection '{collection_name}' 不存在",
            }
            continue

        collection = Collection(name=collection_name)
        collection.load()

        def get_dataset_count(dataset_name: str) -> Dict:
            """获取数据集在 collection 中的记录数。"""
            try:
                results = collection.query(
                    expr=f'dataset_name == "{dataset_name}"',
                    output_fields=["id"],
                    limit=1,
                )

                if not results:
                    return {"count": 0, "count_str": "0", "has_data": False}

                # Milvus 的 query limit 最大值为 16384
                max_query_limit = 16384
                all_results = collection.query(
                    expr=f'dataset_name == "{dataset_name}"',
                    output_fields=["id"],
                    limit=max_query_limit,
                )
                count = len(all_results)

                if count >= max_query_limit:
                    count_str = f">= {count:,} (可能更多，已限制查询)"
                else:
                    count_str = f"{count:,}"

                return {
                    "count": count,
                    "count_str": count_str,
                    "has_data": count > 0,
                }
            except Exception as e:
                return {"count": 0, "count_str": f"错误: {str(e)}", "has_data": False, "error": str(e)}

        dataset1_info = get_dataset_count(dataset_name1)
        dataset2_info = get_dataset_count(dataset_name2)

        comparison[collection_type.value] = {
            "collection_name": collection_name,
            "exists": True,
            "dataset1": dataset1_info,
            "dataset2": dataset2_info,
        }

    return comparison


def print_datasets_comparison(dataset_name1: str, dataset_name2: str):
    """以表格形式打印两个数据集在 4 个 collection 中的对比情况。"""
    print("=" * 100)
    print(f"数据集对比: '{dataset_name1}' vs '{dataset_name2}'")
    print("=" * 100)

    comparison = compare_datasets(dataset_name1, dataset_name2)

    # 打印表头
    print(f"\n{'Collection 类型':<25} {'Collection 名称':<30} {dataset_name1:<20} {dataset_name2:<20}")
    print("-" * 100)

    # 打印每个 collection 的对比信息
    for collection_type, data in comparison.items():
        collection_name = data.get("collection_name", "N/A")
        
        if not data.get("exists"):
            print(f"{collection_type:<25} {collection_name:<30} {'Collection 不存在':<20} {'Collection 不存在':<20}")
            continue

        dataset1_count = data.get("dataset1", {}).get("count_str", "0")
        dataset2_count = data.get("dataset2", {}).get("count_str", "0")

        # 如果有错误，显示错误信息
        if "error" in data.get("dataset1", {}):
            error_msg = data['dataset1']['error']
            # 提取错误的关键信息
            if "invalid max query result window" in error_msg:
                dataset1_count = "查询限制错误"
            else:
                dataset1_count = f"错误: {error_msg[:30]}..."
        if "error" in data.get("dataset2", {}):
            error_msg = data['dataset2']['error']
            # 提取错误的关键信息
            if "invalid max query result window" in error_msg:
                dataset2_count = "查询限制错误"
            else:
                dataset2_count = f"错误: {error_msg[:30]}..."

        print(f"{collection_type:<25} {collection_name:<30} {dataset1_count:<20} {dataset2_count:<20}")

    print("-" * 100)
    
    # 打印汇总信息
    print("\n汇总:")
    total1 = sum(
        data.get("dataset1", {}).get("count", 0)
        for data in comparison.values()
        if data.get("exists") and "error" not in data.get("dataset1", {})
    )
    total2 = sum(
        data.get("dataset2", {}).get("count", 0)
        for data in comparison.values()
        if data.get("exists") and "error" not in data.get("dataset2", {})
    )
    print(f"  {dataset_name1} 总记录数: {total1:,}")
    print(f"  {dataset_name2} 总记录数: {total2:,}")
    diff = total1 - total2
    if diff > 0:
        print(f"  差异: {abs(diff):,} ({dataset_name1} 多 {abs(diff):,} 条)")
    elif diff < 0:
        print(f"  差异: {abs(diff):,} ({dataset_name2} 多 {abs(diff):,} 条)")
    else:
        print(f"  差异: 0 (两个数据集记录数相同)")


def main():
    """主函数，用于命令行交互。"""
    import sys

    if len(sys.argv) < 2:
        print("用法:")
        print("  python check_milvus.py list                    # 列出所有 collection")
        print("  python check_milvus.py info <dataset_name>      # 查询数据集信息")
        print("  python check_milvus.py compare <dataset1> <dataset2>  # 对比两个数据集")
        print("  python check_milvus.py delete <dataset_name>    # 删除数据集（需要确认）")
        print("  python check_milvus.py delete <dataset_name> --confirm  # 确认删除")
        return

    command = sys.argv[1]

    if command == "list":
        print_all_collections()
    elif command == "info":
        if len(sys.argv) < 3:
            print("错误: 请提供数据集名称")
            return
        dataset_name = sys.argv[2]
        print_dataset_info(dataset_name)
    elif command == "compare":
        if len(sys.argv) < 4:
            print("错误: 请提供两个数据集名称")
            print("用法: python check_milvus.py compare <dataset1> <dataset2>")
            return
        dataset_name1 = sys.argv[2]
        dataset_name2 = sys.argv[3]
        print_datasets_comparison(dataset_name1, dataset_name2)
    elif command == "delete":
        if len(sys.argv) < 3:
            print("错误: 请提供数据集名称")
            return
        dataset_name = sys.argv[2]
        confirm = "--confirm" in sys.argv or "-y" in sys.argv

        if not confirm:
            print(f"警告: 即将删除数据集 '{dataset_name}' 在所有 collection 中的数据")
            print("请使用 --confirm 或 -y 参数确认删除")
            return

        print(f"正在删除数据集 '{dataset_name}' 的所有数据...")
        results = delete_dataset_from_all_collections(dataset_name, confirm=True)
        print("\n删除结果:")
        for collection_type, success in results.items():
            status = "成功" if success else "失败"
            print(f"  {collection_type}: {status}")
    else:
        print(f"未知命令: {command}")


if __name__ == "__main__":
    main()
    
    # 使用示例:
    # python3 check_milvus.py list
    # python3 check_milvus.py info electrolytes_highlevel
    # python3 check_milvus.py info paper_mini_highlevel
    # python3 check_milvus.py compare electrolytes_highlevel paper_mini_highlevel
    # 对比两个数据集在 4 个 collection 中的情况
    # python3 check_milvus.py compare electrolytes_highlevel paper_mini_highlevel
    
    
    # 删除 electrolytes_highlevel 数据集（需要确认）
    # python3 check_milvus.py delete electrolytes_highlevel --confirm
    
    # [2025-11-27 11:37:50] INFO     milvus:78 - Milvus 连接成功: 127.0.0.1:19530
    # 成功从 collection 'graph_node_names' 中删除数据集 'electrolytes_highlevel'
    # 成功从 collection 'graph_triple_strings' 中删除数据集 'electrolytes_highlevel'
    # 成功从 collection 'graph_schema_names' 中删除数据集 'electrolytes_highlevel'
    # 成功从 collection 'graph_community_names' 中删除数据集 'electrolytes_highlevel'

    # 删除结果:
    # node_names: 成功
    # triple_strings: 成功
    # schema_names: 成功
    # community_names: 成功
    
