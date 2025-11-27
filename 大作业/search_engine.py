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
import jieba
#使用国内的镜像hugging face网站
#国内的也炸了，迫不得已上梯子


# 导入sentence-transformers库
from sentence_transformers import SentenceTransformer, util

class SearchEngine:
    """搜索引擎核心类"""
    #可使用本地模型，只需将str替换即可def __init__(self, model_name: str = '/Users/wanglijiahang/Downloads/大作业/local_model'):，默认是联网的
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
            
            # 使用jieba分词，支持中英文混合分词
            words = set(jieba.cut(text))
            
            # 同时保留英文单词的split分词（处理连续英文）
            words.update(text.split())
            
            for word in words:
                # 移除空格和空字符串
                word = word.strip()
                if not word or len(word) <= 1:  # 调整阈值以保留更多有意义的中文单字词
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
    
    def expand_query(self, query: str) -> str:
        """
        查询扩展，将相关术语作为补充上下文添加到原始查询中
        
        Args:
            query: 清洗后的查询
            
        Returns:
            扩展后的上下文文本
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
        
        # 基础查询始终保留
        expanded_context = [query]
        
        # 收集扩展术语，但限制数量以避免噪音过大
        collected_terms = set()
        
        # 检查是否包含中文关键词需要扩展
        for chinese_term, english_terms in expansions.items():
            if chinese_term in query:
                # 只添加相关的核心术语（限制数量）
                for term in english_terms[:5]:  # 只取前5个最相关的术语
                    if term not in collected_terms:
                        collected_terms.add(term)
                        expanded_context.append(term)
        
        # 将扩展后的上下文连接成一个文本
        return " ".join(expanded_context)
    
    def _calculate_similarity(self, query_vector: np.ndarray) -> List[Tuple[int, float]]:
        """
        计算查询向量与所有文档向量的相似度
        
        Args:
            query_vector: 查询向量
            
        Returns:
            文档ID和相似度分数的列表
        """
        # 使用sentence-transformers的util.cos_sim进行高效的矩阵计算
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # 计算所有文档向量与查询向量的余弦相似度（矩阵运算，速度极快）
        similarities = util.cos_sim(query_vector, self.document_vectors)[0]
        
        # 转换为(索引, 分数)的列表
        similarities_list = [(i, similarities[i].item()) for i in range(len(similarities))]
        
        # 按相似度降序排序
        similarities_list.sort(key=lambda x: x[1], reverse=True)
        
        return similarities_list
    
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
        
        # 使用jieba分词查询文本，支持中英文混合查询
        query_words = set(jieba.cut(cleaned_query))
        
        # 同时保留英文单词的split分词
        query_words.update(cleaned_query.split())
        
        # 计算文档匹配分数
        doc_scores = Counter()
        
        for word in query_words:
            # 移除空格和空字符串
            word = word.strip()
            if not word or len(word) <= 1:
                continue
                
            if word in self.keyword_index:
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
    
    def search(self, query: str, top_k: int = 10, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """
        混合搜索（语义搜索 + 关键词搜索），使用RRF算法融合结果
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            rrf_k: RRF常数，默认60
            
        Returns:
            搜索结果列表
        """
        print(f"搜索查询: '{query}'")
        
        # 清洗查询
        cleaned_query = self.clean_query(query)
        
        # 获取扩展上下文
        expanded_context = self.expand_query(cleaned_query)
        
        # 语义搜索：使用扩展上下文进行单次搜索，获取更多结果用于RRF
        semantic_results = self.semantic_search(expanded_context, top_k=top_k*2)
        semantic_scores = {}
        for doc_idx, score in semantic_results:
            doc_id = self.processed_data[doc_idx]['id']
            semantic_scores[doc_id] = score
        
        # 关键词搜索：使用原始查询进行搜索，获取更多结果用于RRF
        # 1. 原始查询的关键词搜索
        keyword_scores = self.keyword_search(cleaned_query, top_k=top_k*2)
        
        # 2. 如果查询中有特定术语，为相关关键词增加权重
        expanded_terms = expanded_context.split()
        # 提取扩展术语中的核心词（排除原始查询词）
        query_terms = set(cleaned_query.split())
        additional_terms = [term for term in expanded_terms if term.lower() not in query_terms]
        
        # 为包含额外术语的文档增加权重
        for term in additional_terms:
            term = term.strip().lower()
            if term in self.keyword_index:
                for doc_id in self.keyword_index[term]:
                    # 增加权重但保持在合理范围内
                    if doc_id in keyword_scores:
                        keyword_scores[doc_id] = min(1.0, keyword_scores[doc_id] * 1.2)
        
        # 为语义搜索结果构建排名映射
        semantic_ranks = {}
        for rank, (doc_id, _) in enumerate(sorted(semantic_scores.items(), key=lambda x: x[1], reverse=True)):
            semantic_ranks[doc_id] = rank + 1  # 排名从1开始
        
        # 为关键词搜索结果构建排名映射
        keyword_ranks = {}
        for rank, (doc_id, _) in enumerate(sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)):
            keyword_ranks[doc_id] = rank + 1  # 排名从1开始
        
        # 应用RRF算法融合结果
        rrf_scores = {}
        
        # 确保两个结果集的文档ID都被考虑
        all_doc_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
        
        for doc_id in all_doc_ids:
            # 获取文档在语义搜索中的排名（如果没有则排名为无穷大）
            semantic_rank = semantic_ranks.get(doc_id, float('inf'))
            
            # 获取文档在关键词搜索中的排名（如果没有则排名为无穷大）
            keyword_rank = keyword_ranks.get(doc_id, float('inf'))
            
            # 计算RRF分数: 1/(k + rank)
            rrf_score = 0
            if semantic_rank != float('inf'):
                rrf_score += 1 / (rrf_k + semantic_rank)
            if keyword_rank != float('inf'):
                rrf_score += 1 / (rrf_k + keyword_rank)
            
            # 保存RRF分数
            rrf_scores[doc_id] = rrf_score
        
        # 按RRF分数降序排序
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 构建最终结果
        final_results = []
        for doc_id, rrf_score in sorted_results[:top_k]:
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
                    'score': rrf_score,
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
