#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNESCO项目搜索引擎核心模块
实现模糊搜索和语义匹配功能
"""

import os
import re
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter
#使用国内的镜像hugging face网站
#国内的也炸了，迫不得已上梯子


# 导入sentence-transformers库
from sentence_transformers import SentenceTransformer

class SearchEngine:
    """搜索引擎核心类"""
    
    def __init__(self, model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        """
        初始化搜索引擎
        
        Args:
            model_name: 使用的预训练模型名称
        """
        self.model_name = model_name
        self.model = None
        self.processed_data = None
        self.keyword_index = None
        self.document_vectors = None
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """
        加载NLP模型
        """
        print(f"正在加载预训练模型: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            print("模型加载成功")
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            raise
            
    
    def load_data(self, data_path: str, index_path: str = None):
        """
        加载预处理数据和索引
        
        Args:
            data_path: 预处理数据路径
            index_path: 索引文件路径
        """
        print(f"正在加载处理后的数据: {data_path}")
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.processed_data = json.load(f)
            print(f"成功加载 {len(self.processed_data)} 条记录")
        except Exception as e:
            print(f"加载数据时出错: {str(e)}")
            raise
        
        # 加载关键词索引
        if index_path and os.path.exists(index_path):
            print(f"正在加载关键词索引: {index_path}")
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    self.keyword_index = json.load(f)
                print(f"关键词索引加载成功，包含 {len(self.keyword_index)} 个关键词")
            except Exception as e:
                print(f"加载索引时出错: {str(e)}")
                print("将重新构建关键词索引")
                self.keyword_index = self._build_keyword_index()
        else:
            print("未找到关键词索引，将重新构建")
            self.keyword_index = self._build_keyword_index()
        
        # 生成文档向量
        self._generate_document_vectors()
    
    def _build_keyword_index(self) -> Dict[str, List[int]]:
        """
        构建关键词索引
        
        Returns:
            关键词到文档ID的映射
        """
        keyword_index = {}
        
        for record in self.processed_data:
            doc_id = record['id']
            text = record['raw_text']
            
            # 简单分词
            words = set(text.split())
            
            for word in words:
                if len(word) <= 2:
                    continue
                
                if word not in keyword_index:
                    keyword_index[word] = []
                keyword_index[word].append(doc_id)
        
        return keyword_index
    
    def _generate_document_vectors(self):
        """
        生成文档向量
        """
        if self.model is not None:
            print("正在生成文档向量...")
            texts = [record['raw_text'] for record in self.processed_data]
            self.document_vectors = self.model.encode(texts, show_progress_bar=True)
            print(f"文档向量生成完成，形状: {self.document_vectors.shape}")
        else:
            raise ValueError("模型未正确加载，无法生成文档向量")
    

    
    def clean_query(self, query: str) -> str:
        """
        清洗查询文本
        
        Args:
            query: 原始查询
            
        Returns:
            清洗后的查询
        """
        # 转换为小写
        query = query.lower()
        
        # 移除特殊字符
        query = re.sub(r'[^\w\s.,;!?\-()]', '', query)
        
        # 移除多余空格
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query
    
    def expand_query(self, query: str) -> List[str]:
        """
        查询扩展，处理口语化表达
        
        Args:
            query: 清洗后的查询
            
        Returns:
            扩展后的查询列表
        """
        # 常见口语表达映射到正式术语
        expansions = {
    '教育项目': [
        # 原有基础
        'education project', 'educational program', 'learning initiative',
        # UN 官方术语 (SDG 4 相关)
        'inclusive education', 'equitable quality education', 'lifelong learning',
        'early childhood development', 'technical and vocational', 'TVET',
        'universal literacy', 'numeracy', 'capacity-building', 'pedagogical',
        'non-formal education', 'scholastic', 'tertiary', 'secondary',
        'vocational training', 'skills development', 'access to education'
    ],
    '文化遗产': [
        # 原有基础
        'cultural heritage', 'heritage conservation', 'cultural preservation',
        # UN/UNESCO 官方术语
        'tangible heritage', 'intangible cultural heritage', 'world heritage',
        'underwater cultural heritage', 'intercultural dialogue', 'cultural diversity',
        'restoration', 'safeguarding', 'archaeological', 'historical',
        'indigenous knowledge', 'traditional craftsmanship', 'museum management',
        'illicit trafficking'  # 联合国常涉及文物非法贩运议题
    ],
    '气候变化': [
        # 原有基础
        'climate change', 'global warming', 'climate action',
        # UN/UNFCCC 官方术语 (SDG 13 & 巴黎协定)
        'climate mitigation', 'climate adaptation', 'climate resilience',
        'low-carbon', 'net-zero emissions', 'greenhouse gas', 'GHG',
        'anthropogenic', 'meteorological', 'extreme weather', 'disaster risk reduction',
        'carbon footprint', 'renewable energy', 'nationally determined contributions', 'NDC',
        'environmental sustainability', 'biodiversity loss'
    ],
    '水资源': [
        # 原有基础
        'water resources', 'water management', 'water conservation',
        # UN 官方术语 (SDG 6)
        'clean water and sanitation', 'WASH', 'potable water', 'wastewater management',
        'transboundary water', 'integrated water resources management', 'IWRM',
        'hydrological', 'aquatic ecosystems', 'water scarcity', 'water stress',
        'hygiene', 'sanitary', 'freshwater', 'marine resources'
    ],
    '非洲': [
        # 原有基础
        'africa', 'regional africa',
        # UN 地缘政治术语
        'sub-saharan africa', 'north africa', 'west africa', 'east africa',
        'central africa', 'southern africa', 'pan-african', 'african union', 'AU',
        'least developed countries', 'LDCs'  # 非洲许多国家属于此列
    ],
    '亚洲': [
        # 原有基础
        'asia', 'asia and the pacific', 'eastern asia',
        # UN 地缘政治术语
        'asia-pacific', 'ESCAP region', 'central asia', 'south asia',
        'southeast asia', 'western asia', 'asean', 'transcontinental'
    ],
    '欧洲': [
        # 原有基础
        'europe', 'western europe', 'eastern europe',
        # UN 地缘政治术语
        'UNECE region', 'european union', 'EU', 'northern europe',
        'southern europe', 'western balkans', 'transition economies'
    ],
    '可持续发展': [
        # 原有基础
        'sustainable development', 'sustainability', 'sdg',
        # UN 核心纲领术语 (Agenda 2030)
        '2030 agenda', 'global goals', 'inclusive growth', 'green economy',
        'circular economy', 'socio-economic', 'environmental', 'multilateral',
        'cross-cutting', 'holistic approach', 'viable', 'equitable',
        'leave no one behind'  # 联合国核心承诺
    ],
    '性别平等': [
        # 原有基础
        'gender equality', 'gender equity', 'women empowerment',
        # UN 官方术语 (SDG 5 & UN Women)
        'gender parity', 'gender mainstreaming', 'gender-based', 'GBV',
        'advancement of women', 'reproductive health', 'feminist',
        'non-discriminatory', 'equal opportunity', 'sexual and reproductive rights',
        'disaggregated data'  # 联合国统计中强调按性别分列的数据
    ],
    '青年': [
        # 原有基础
        'youth', 'young people', 'adolescents',
        # UN 官方术语
        'youth empowerment', 'youth-led', 'intergenerational', 'juvenile',
        'demographic dividend', 'future generations', 'young adults',
        'youth participation', 'civic engagement'
    ]
}
        
        expanded_queries = [query]  # 保留原始查询
        
        # 检查是否包含中文关键词需要扩展
        for chinese_term, english_terms in expansions.items():
            if chinese_term in query:
                expanded_queries.extend(english_terms)
        
        return expanded_queries
    
    def _calculate_similarity(self, query_vector: np.ndarray) -> List[Tuple[int, float]]:
        """
        计算查询向量与所有文档向量的相似度
        
        Args:
            query_vector: 查询向量
            
        Returns:
            文档ID和相似度分数的列表
        """
        similarities = []
        
        # 计算余弦相似度
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        for i, doc_vector in enumerate(self.document_vectors):
            # 归一化向量
            doc_norm = np.linalg.norm(doc_vector)
            query_norm = np.linalg.norm(query_vector)
            
            if doc_norm > 0 and query_norm > 0:
                # 余弦相似度
                similarity = np.dot(doc_vector, query_vector[0]) / (doc_norm * query_norm)
            else:
                similarity = 0
            
            similarities.append((i, similarity))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities
    
    def keyword_search(self, query: str, top_k: int = 50) -> Dict[int, float]:
        """
        关键词搜索
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            
        Returns:
            文档ID到相关性分数的映射
        """
        cleaned_query = self.clean_query(query)
        query_words = set(cleaned_query.split())
        
        # 计算文档匹配分数
        doc_scores = Counter()
        
        for word in query_words:
            if len(word) > 2 and word in self.keyword_index:
                # 为包含该词的文档加分
                for doc_id in self.keyword_index[word]:
                    doc_scores[doc_id] += 1
        
        # 归一化分数
        if doc_scores:
            max_score = max(doc_scores.values())
            for doc_id in doc_scores:
                doc_scores[doc_id] = doc_scores[doc_id] / max_score
        
        # 转换为字典并返回
        return dict(doc_scores.most_common(top_k))
    
    def semantic_search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """
        语义搜索
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            
        Returns:
            文档ID和相似度分数的列表
        """
        cleaned_query = self.clean_query(query)
        
        # 使用预训练模型编码查询
        query_vector = self.model.encode([cleaned_query])[0]
        
        # 计算相似度
        similarities = self._calculate_similarity(query_vector)
        
        # 返回前k个结果
        return similarities[:top_k]
    
    def search(self, query: str, top_k: int = 10, hybrid_weight: float = 0.7) -> List[Dict[str, Any]]:
        """
        混合搜索（语义搜索 + 关键词搜索）
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            hybrid_weight: 语义搜索权重 (0-1)
            
        Returns:
            搜索结果列表
        """
        print(f"搜索查询: '{query}'")
        
        # 扩展查询
        expanded_queries = self.expand_query(query)
        
        # 语义搜索分数
        semantic_scores = {}
        for expanded_query in expanded_queries:
            results = self.semantic_search(expanded_query, top_k=100)
            for doc_idx, score in results:
                doc_id = self.processed_data[doc_idx]['id']
                if doc_id not in semantic_scores or score > semantic_scores[doc_id]:
                    semantic_scores[doc_id] = score
        
        # 关键词搜索分数
        keyword_scores = {}
        for expanded_query in expanded_queries:
            results = self.keyword_search(expanded_query, top_k=100)
            for doc_id, score in results.items():
                if doc_id not in keyword_scores or score > keyword_scores[doc_id]:
                    keyword_scores[doc_id] = score
        
        # 合并分数
        all_doc_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
        combined_scores = {}
        
        for doc_id in all_doc_ids:
            s_score = semantic_scores.get(doc_id, 0)
            k_score = keyword_scores.get(doc_id, 0)
            
            # 混合分数计算
            if s_score > 0 and k_score > 0:
                # 当两种搜索都有结果时，使用加权平均
                combined_score = hybrid_weight * s_score + (1 - hybrid_weight) * k_score
            elif s_score > 0:
                combined_score = s_score * 0.9  # 降低纯语义搜索的置信度
            else:
                combined_score = k_score * 0.8  # 降低纯关键词搜索的置信度
            
            combined_scores[doc_id] = combined_score
        
        # 按分数排序
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 构建最终结果
        final_results = []
        for doc_id, score in sorted_results[:top_k]:
            # 查找文档索引
            doc_idx = None
            for i, record in enumerate(self.processed_data):
                if record['id'] == doc_id:
                    doc_idx = i
                    break
            
            if doc_idx is not None:
                record = self.processed_data[doc_idx]
                result = {
                    'id': doc_id,
                    'score': score,
                    'project': record['original_row']['Project'],
                    'description': record['original_row']['Description'],
                    'sector': record['original_row'].get('Sector', ''),
                    'regional_group': record['original_row'].get('Regional Group', ''),
                    'coordinates': record['original_row'].get('Coordinates', ''),
                    'metadata': record['metadata']
                }
                final_results.append(result)
        
        print(f"找到 {len(final_results)} 个相关结果")
        return final_results
    
    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """
        格式化搜索结果
        
        Args:
            results: 搜索结果列表
            
        Returns:
            格式化的文本
        """
        formatted_text = "\n===== 搜索结果 =====\n\n"
        
        if not results:
            return formatted_text + "未找到相关结果\n"
        
        for i, result in enumerate(results, 1):
            formatted_text += f"【结果 {i}】\n"
            formatted_text += f"项目名称: {result['project']}\n"
            formatted_text += f"类别: {result['sector']}\n"
            formatted_text += f"区域: {result['regional_group']}\n"
            formatted_text += f"坐标: {result['coordinates']}\n"
            
            # 截断过长的描述
            description = result['description']
            if len(description) > 200:
                description = description[:200] + "..."
            formatted_text += f"描述: {description}\n"
            
            formatted_text += f"相关度: {result['score']:.4f}\n"
            formatted_text += "-" * 50 + "\n\n"
        
        return formatted_text

def main():
    """主函数，用于测试"""
    # 初始化搜索引擎
    engine = SearchEngine()
    
    # 加载数据
    data_path = '/Users/wanglijiahang/Downloads/大作业/processed_unesco_data.json'
    index_path = '/Users/wanglijiahang/Downloads/大作业/cache/keyword_index.json'
    
    try:
        engine.load_data(data_path, index_path)
        
        # 测试搜索
        test_queries = [
            "教育项目",
            "非洲文化遗产保护",
            "气候变化应对"
        ]
        
        for query in test_queries:
            print(f"\n\n测试查询: '{query}'")
            results = engine.search(query, top_k=3)
            print(engine.format_results(results))
            
    except Exception as e:
        print(f"搜索过程中发生错误: {str(e)}")

if __name__ == "__main__":
    main()
