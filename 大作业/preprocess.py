#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNESCO项目数据预处理模块
负责数据加载、清洗、文本规范化和特征提取
"""

import os
import re
import json
import pandas as pd
from typing import List, Dict, Any

class DataPreprocessor:
    """数据预处理类"""
    
    def __init__(self):
        """初始化预处理类"""
        # 定义需要处理的重要字段及其权重
        self.important_fields = {
            'Project': 0.4,          # 项目名称 - 最高权重
            'Description': 0.3,      # 项目描述 - 高权重
            'Outcome EN': 0.15,      # 成果说明 - 中等权重
            'Output EN': 0.15        # 产出说明 - 中等权重
        }
        
        # 定义元数据字段
        self.metadata_fields = [
            'Sector', 'Beneficiary', 'Donor', 
            'Beneficiary Type', 'Regional Group', 
            'Coordinates', 'Start Date', 'End Date'
        ]
        
        # 缓存目录
        self.cache_dir = 'cache'
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        加载CSV数据
        
        Args:
            file_path: CSV文件路径
            
        Returns:
            加载的DataFrame对象
        """
        print(f"正在加载数据: {file_path}")
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path, encoding='utf-8')
            print(f"成功加载 {len(df)} 条记录")
            return df
        except Exception as e:
            print(f"加载数据时出错: {str(e)}")
            raise
    
    def clean_text(self, text: Any) -> str:
        """
        清洗和规范化文本
        
        Args:
            text: 输入文本
            
        Returns:
            清洗后的文本
        """
        if pd.isna(text):
            return ""
        
        # 转换为字符串
        text = str(text)
        
        # 移除多余空格和换行符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符，保留字母、数字、空格和基本标点
        text = re.sub(r'[^\w\s.,;!?\-()]', '', text)
        
        # 标准化为小写
        text = text.lower()
        
        # 移除首尾空格
        text = text.strip()
        
        return text
    
    def create_unified_text(self, row: pd.Series) -> str:
        """
        创建统一的文本表示，合并多个重要字段
        
        Args:
            row: 数据行
            
        Returns:
            合并后的文本
        """
        texts = []
        
        # 按照权重合并文本
        for field, weight in self.important_fields.items():
            if field in row and not pd.isna(row[field]):
                # 根据权重重复文本，简单的TF增强方式
                # 注意：这里只是概念展示，实际向量表示中权重会在检索时应用
                text = self.clean_text(row[field])
                if text:
                    texts.append(text)
        
        # 合并所有文本，用特殊标记分隔
        unified_text = " | ".join(texts)
        return unified_text
    
    def extract_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """
        提取元数据
        
        Args:
            row: 数据行
            
        Returns:
            元数据字典
        """
        metadata = {}
        
        for field in self.metadata_fields:
            if field in row:
                value = row[field]
                if not pd.isna(value):
                    metadata[field] = str(value).strip()
                else:
                    metadata[field] = None
        
        return metadata
    
    def preprocess_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        预处理整个数据集
        
        Args:
            df: 输入DataFrame
            
        Returns:
            预处理后的记录列表
        """
        print("开始预处理数据...")
        processed_records = []
        
        for idx, row in df.iterrows():
            # 创建统一文本表示
            unified_text = self.create_unified_text(row)
            
            # 提取元数据
            metadata = self.extract_metadata(row)
            
            # 构建记录
            record = {
                'id': idx,
                'project_name': self.clean_text(row.get('Project', '')),
                'raw_text': unified_text,
                'metadata': metadata,
                'original_row': row.to_dict()  # 保存原始数据用于展示
            }
            
            processed_records.append(record)
            
            # 显示进度
            if (idx + 1) % 100 == 0:
                print(f"已处理 {idx + 1}/{len(df)} 条记录")
        
        print(f"数据预处理完成，共处理 {len(processed_records)} 条记录")
        return processed_records
    
    def save_processed_data(self, data: List[Dict[str, Any]], output_path: str) -> None:
        """
        保存预处理后的数据
        
        Args:
            data: 预处理后的数据
            output_path: 输出文件路径
        """
        print(f"正在保存预处理后的数据到: {output_path}")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"数据保存成功")
        except Exception as e:
            print(f"保存数据时出错: {str(e)}")
            raise
    
    def load_processed_data(self, input_path: str) -> List[Dict[str, Any]]:
        """
        加载预处理后的数据
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            预处理后的数据
        """
        print(f"正在加载预处理后的数据: {input_path}")
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"成功加载 {len(data)} 条记录")
            return data
        except Exception as e:
            print(f"加载数据时出错: {str(e)}")
            raise
    
    def build_keyword_index(self, data: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """
        构建简单的关键词索引
        
        Args:
            data: 预处理后的数据
            
        Returns:
            关键词到文档ID的映射
        """
        print("正在构建关键词索引...")
        keyword_index = {}
        
        for record in data:
            doc_id = record['id']
            text = record['raw_text']
            
            # 简单分词（按空格）
            words = set(text.split())
            
            for word in words:
                # 忽略太短的词
                if len(word) <= 2:
                    continue
                
                if word not in keyword_index:
                    keyword_index[word] = []
                keyword_index[word].append(doc_id)
        
        print(f"关键词索引构建完成，包含 {len(keyword_index)} 个关键词")
        return keyword_index
    
    def save_index(self, index: Any, index_name: str) -> None:
        """
        保存索引到缓存
        
        Args:
            index: 索引对象
            index_name: 索引名称
        """
        index_path = os.path.join(self.cache_dir, f"{index_name}.json")
        print(f"正在保存索引到: {index_path}")
        try:
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
            print("索引保存成功")
        except Exception as e:
            print(f"保存索引时出错: {str(e)}")
            raise

def main():
    """主函数"""
    # CSV文件路径
    csv_file = '/Users/wanglijiahang/Downloads/大作业/UNESCO_projects_coordinates_completed.csv'
    # 预处理后的数据输出路径
    processed_file = '/Users/wanglijiahang/Downloads/大作业/processed_unesco_data.json'
    
    # 创建预处理器实例
    preprocessor = DataPreprocessor()
    
    try:
        # 加载数据
        df = preprocessor.load_data(csv_file)
        
        # 预处理数据
        processed_data = preprocessor.preprocess_data(df)
        
        # 保存预处理后的数据
        preprocessor.save_processed_data(processed_data, processed_file)
        
        # 构建关键词索引
        keyword_index = preprocessor.build_keyword_index(processed_data)
        preprocessor.save_index(keyword_index, 'keyword_index')
        
        print("\n数据预处理和索引构建全部完成！")
        print(f"预处理后的数据保存至: {processed_file}")
        print(f"索引文件保存至缓存目录: {preprocessor.cache_dir}")
        
    except Exception as e:
        print(f"预处理过程中发生错误: {str(e)}")

if __name__ == "__main__":
    main()
