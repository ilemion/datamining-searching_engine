#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNESCO项目搜索引擎 - 测试和优化脚本
用于测试功能和性能，提供优化建议
"""

import os
import time
import json
import numpy as np
import sys
from tqdm import tqdm
from typing import List, Dict, Any, Tuple

# 导入搜索引擎模块
sys.path.append('/Users/wanglijiahang/Downloads/大作业')
from search_engine import SearchEngine

class SearchTester:
    """搜索引擎测试类"""
    
    def __init__(self):
        """初始化测试类"""
        # 文件路径
        self.data_path = '/Users/wanglijiahang/Downloads/大作业/processed_unesco_data.json'
        self.index_path = '/Users/wanglijiahang/Downloads/大作业/cache/keyword_index.json'
        
        # 测试查询集
        self.test_queries = [
            # 中文查询
            "教育项目",
            "文化遗产保护",
            "非洲水资源管理",
            "气候变化应对",
            "性别平等项目",
            # 英文查询
            "cultural heritage conservation",
            "education in Asia",
            "climate change adaptation",
            "water resource management",
            # 混合查询
            "非洲 climate change",
            "教育 education",
            # 模糊查询
            "非遗保护",  # 非物质文化遗产
            "可持续发展目标",  # SDG
            "青年参与"
        ]
        
        # 搜索引擎实例
        self.engine = None
        
        # 测试结果
        self.test_results = {}
    
    def initialize_engine(self):
        """初始化搜索引擎"""
        print("初始化搜索引擎...")
        try:
            self.engine = SearchEngine()
            self.engine.load_data(self.data_path, self.index_path)
            print("搜索引擎初始化成功！")
            return True
        except Exception as e:
            print(f"初始化失败: {str(e)}")
            return False
    
    def test_search_performance(self):
        """测试搜索性能"""
        print("\n===== 性能测试 =====")
        
        total_time = 0
        results_count = []
        
        for query in tqdm(self.test_queries, desc="测试查询"):
            start_time = time.time()
            results = self.engine.search(query, top_k=10)
            elapsed_time = time.time() - start_time
            
            # 记录结果
            self.test_results[query] = {
                'time': elapsed_time,
                'results_count': len(results),
                'top_result_score': results[0]['score'] if results else 0
            }
            
            total_time += elapsed_time
            results_count.append(len(results))
            
            # 记录详细结果
            if len(results) > 0:
                self.test_results[query]['top_project'] = results[0]['project']
        
        # 计算统计信息
        avg_time = total_time / len(self.test_queries)
        avg_results = np.mean(results_count)
        
        print(f"\n性能统计:")
        print(f"平均搜索时间: {avg_time:.4f} 秒")
        print(f"平均结果数量: {avg_results:.1f}")
        print(f"总查询数: {len(self.test_queries)}")
        print(f"总耗时: {total_time:.2f} 秒")
        
        return avg_time, avg_results
    
    def test_query_types(self):
        """测试不同类型的查询"""
        print("\n===== 查询类型测试 =====")
        
        query_types = {
            '中文查询': [q for q in self.test_queries if any('\u4e00' <= char <= '\u9fff' for char in q)],
            '英文查询': [q for q in self.test_queries if all(ord(char) < 128 for char in q)],
            '模糊查询': ["非遗保护", "可持续发展目标", "青年参与"]
        }
        
        for query_type, queries in query_types.items():
            print(f"\n{query_type}:")
            for query in queries:
                if query in self.test_results:
                    result = self.test_results[query]
                    print(f"  '{query}': 耗时={result['time']:.4f}s, 结果数={result['results_count']}")
                    if 'top_project' in result:
                        project = result['top_project'][:50] + '...' if len(result['top_project']) > 50 else result['top_project']
                        print(f"    最佳匹配: {project}")
    
    def evaluate_result_quality(self):
        """评估结果质量"""
        print("\n===== 结果质量评估 =====")
        
        high_score_queries = []
        low_score_queries = []
        empty_result_queries = []
        
        for query, result in self.test_results.items():
            if result['results_count'] == 0:
                empty_result_queries.append(query)
            elif result['top_result_score'] > 0.7:
                high_score_queries.append((query, result['top_result_score']))
            elif result['top_result_score'] < 0.3:
                low_score_queries.append((query, result['top_result_score']))
        
        print(f"高相关性结果 (>0.7) 的查询数: {len(high_score_queries)}")
        for query, score in high_score_queries[:5]:  # 显示前5个
            print(f"  '{query}': {score:.4f}")
        
        print(f"\n低相关性结果 (<0.3) 的查询数: {len(low_score_queries)}")
        for query, score in low_score_queries[:5]:
            print(f"  '{query}': {score:.4f}")
        
        print(f"\n无结果的查询数: {len(empty_result_queries)}")
        if empty_result_queries:
            for query in empty_result_queries:
                print(f"  '{query}'")
    
    def save_test_results(self):
        """保存测试结果"""
        output_path = '/Users/wanglijiahang/Downloads/大作业/test_results.json'
        print(f"\n保存测试结果到: {output_path}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, ensure_ascii=False, indent=2)
            print("测试结果保存成功")
        except Exception as e:
            print(f"保存测试结果失败: {str(e)}")
    
    def generate_optimization_recommendations(self):
        """生成优化建议"""
        print("\n===== 优化建议 =====")
        
        recommendations = []
        
        # 检查搜索时间
        try:
            avg_time = sum(r['time'] for r in self.test_results.values()) / len(self.test_results)
            if avg_time > 1.0:
                recommendations.append("搜索时间过长，建议优化:")
                recommendations.append("  - 考虑使用更轻量级的模型或TF-IDF")
                recommendations.append("  - 实现向量缓存机制")
                recommendations.append("  - 优化相似度计算算法")
        except (KeyError, ZeroDivisionError):
            recommendations.append("无法计算平均搜索时间，可能没有测试结果")
        
        # 检查结果质量
        low_score_count = sum(1 for r in self.test_results.values() if r['top_result_score'] < 0.3)
        empty_count = sum(1 for r in self.test_results.values() if r['results_count'] == 0)
        
        if low_score_count > len(self.test_results) * 0.3:
            recommendations.append("\n结果相关性较低，建议:")
            recommendations.append("  - 扩展查询扩展词典")
            recommendations.append("  - 调整混合搜索权重")
            recommendations.append("  - 增加更多关键词映射")
        
        if empty_count > 0:
            recommendations.append("\n部分查询无结果，建议:")
            recommendations.append("  - 增加同义词扩展")
            recommendations.append("  - 实现模糊匹配算法")
            recommendations.append("  - 优化查询重写规则")
        
        # 通用优化建议
        recommendations.append("\n通用优化建议:")
        recommendations.append("  - 实现增量索引更新")
        recommendations.append("  - 添加缓存机制减少重复计算")
        recommendations.append("  - 优化内存使用")
        recommendations.append("  - 考虑使用更高效的向量存储库")
        
        # 打印建议
        for rec in recommendations:
            print(rec)
    
    def run_optimization(self):
        """执行简单的优化"""
        print("\n===== 执行基本优化 =====")
        
        # 创建缓存目录
        cache_dir = '/Users/wanglijiahang/Downloads/大作业/cache'
        os.makedirs(cache_dir, exist_ok=True)
        
        # 优化点1: 保存文档向量缓存
        if hasattr(self.engine, 'document_vectors') and self.engine.document_vectors is not None:
            vector_cache_path = os.path.join(cache_dir, 'document_vectors.npy')
            print(f"保存文档向量缓存到: {vector_cache_path}")
            try:
                np.save(vector_cache_path, self.engine.document_vectors)
                print("向量缓存保存成功")
            except Exception as e:
                print(f"保存向量缓存失败: {str(e)}")
        
        # 优化点2: 检查TF-IDF优化
        if not hasattr(self.engine, 'model') or self.engine.model is None:
            print("\nTF-IDF优化建议:")
            print("  - 考虑使用更高效的词汇表过滤")
            print("  - 实现停用词过滤")
            print("  - 添加词干提取或词形还原")
        
        print("\n基本优化完成")
    
    def run_all_tests(self):
        """运行所有测试"""
        if not self.initialize_engine():
            return False
        
        print("开始运行测试...")
        
        # 性能测试
        self.test_search_performance()
        
        # 查询类型测试
        self.test_query_types()
        
        # 结果质量评估
        self.evaluate_result_quality()
        
        # 保存结果
        self.save_test_results()
        
        # 优化建议
        self.generate_optimization_recommendations()
        
        # 执行优化
        self.run_optimization()
        
        print("\n===== 测试和优化完成 =====")
        return True

def update_engine_for_performance(search_engine_path):
    """更新搜索引擎代码以优化性能"""
    try:
        with open(search_engine_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # 添加向量缓存加载功能
        if 'def _load_document_vectors' not in code:
            load_vector_code = '''
    def _load_document_vectors(self):
        """尝试从缓存加载文档向量"""
        vector_cache_path = os.path.join('cache', 'document_vectors.npy')
        if os.path.exists(vector_cache_path):
            print(f"正在从缓存加载文档向量: {vector_cache_path}")
            try:
                self.document_vectors = np.load(vector_cache_path)
                print(f"文档向量加载成功，形状: {self.document_vectors.shape}")
                return True
            except Exception as e:
                print(f"加载向量缓存失败: {str(e)}")
        return False
    
    def _save_document_vectors(self):
        """保存文档向量到缓存"""
        vector_cache_path = os.path.join('cache', 'document_vectors.npy')
        try:
            np.save(vector_cache_path, self.document_vectors)
            print(f"文档向量已保存到缓存: {vector_cache_path}")
            return True
        except Exception as e:
            print(f"保存向量缓存失败: {str(e)}")
            return False'''
            
            # 在_generate_document_vectors方法前插入_load_document_vectors方法
            code = code.replace("    def _generate_document_vectors(self):", 
                               load_vector_code + "\n    def _generate_document_vectors(self):")
            
            # 修改_load_data方法以使用缓存
            code = code.replace("        # 生成文档向量\n        self._generate_document_vectors()", 
                               "        # 尝试加载缓存的文档向量，如果失败则重新生成\n        if not self._load_document_vectors():\n            self._generate_document_vectors()\n            # 保存文档向量到缓存\n            vector_cache_path = os.path.join('cache', 'document_vectors.npy')\n            try:\n                np.save(vector_cache_path, self.document_vectors)\n                print(f'文档向量已保存到缓存: {vector_cache_path}')\n            except Exception as e:\n                print(f'保存向量缓存失败: {str(e)}')")
        
        # 添加NumPy导入
        if 'import numpy as np' not in code:
            # 在文件开头添加导入语句
            code = "import numpy as np\n" + code
        
        # 保存更新后的代码
        with open(search_engine_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        print(f"成功更新 {search_engine_path} 以优化性能")
        return True
    except Exception as e:
        print(f"更新代码失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("UNESCO项目搜索引擎 - 测试和优化工具")
    print("=" * 50)
    
    # 首先更新搜索引擎代码以优化性能
    search_engine_path = '/Users/wanglijiahang/Downloads/大作业/search_engine.py'
    if os.path.exists(search_engine_path):
        update_engine_for_performance(search_engine_path)
    
    # 创建并运行测试器
    tester = SearchTester()
    tester.run_all_tests()
    
    print("\n测试和优化过程已完成！")
    print("建议:")
    print("1. 安装sentence-transformers库以获得更好的语义搜索性能")
    print("2. 运行unesco_search_cli.py体验交互式搜索")
    print("3. 查看test_results.json获取详细测试数据")

if __name__ == "__main__":
    main()