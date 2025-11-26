#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNESCO项目搜索引擎 - 命令行界面
提供交互式查询体验
"""

import os
import sys
import time
import readline  # 用于命令行历史和编辑功能
from datetime import datetime
from typing import List, Optional

# 导入搜索引擎模块
sys.path.append('/Users/wanglijiahang/Downloads/大作业')
from search_engine import SearchEngine
from preprocess import DataPreprocessor

class SearchCLI:
    """命令行界面类"""
    
    def __init__(self):
        """初始化命令行界面"""
        # 配置文件路径
        self.data_path = '/Users/wanglijiahang/Downloads/大作业/processed_unesco_data.json'
        self.index_path = '/Users/wanglijiahang/Downloads/大作业/cache/keyword_index.json'
        self.raw_csv_path = '/Users/wanglijiahang/Downloads/大作业/UNESCO_projects_coordinates_completed.csv'
        
        # 搜索历史
        self.search_history = []
        self.max_history = 20
        
        # 搜索引擎实例
        self.engine = None
        
        # 默认配置
        self.default_top_k = 5
        self.hybrid_weight = 0.7
    
    def setup_readline(self):
        """配置命令行历史"""
        # 设置历史文件
        hist_file = os.path.expanduser('~/.unesco_search_history')
        
        # 尝试读取历史记录
        try:
            readline.read_history_file(hist_file)
            readline.set_history_length(1000)
        except FileNotFoundError:
            pass
        
        # 保存历史记录
        def save_history():
            try:
                readline.write_history_file(hist_file)
            except Exception:
                pass
        
        # 注册退出时保存历史
        import atexit
        atexit.register(save_history)
    
    def initialize_engine(self):
        """初始化搜索引擎"""
        print("正在初始化UNESCO项目搜索引擎...\n")
        
        # 检查预处理数据是否存在，如果不存在则先预处理
        if not os.path.exists(self.data_path):
            print("未找到预处理数据，正在进行数据预处理...\n")
            try:
                preprocessor = DataPreprocessor()
                df = preprocessor.load_data(self.raw_csv_path)
                processed_data = preprocessor.preprocess_data(df)
                preprocessor.save_processed_data(processed_data, self.data_path)
                
                # 构建并保存关键词索引
                keyword_index = preprocessor.build_keyword_index(processed_data)
                preprocessor.save_index(keyword_index, 'keyword_index')
                
                print("数据预处理完成！\n")
            except Exception as e:
                print(f"数据预处理失败: {str(e)}")
                print("请确保CSV文件路径正确并且格式有效。")
                return False
        
        # 初始化搜索引擎
        try:
            self.engine = SearchEngine()
            self.engine.load_data(self.data_path, self.index_path)
            print("搜索引擎初始化完成！\n")
            return True
        except Exception as e:
            print(f"搜索引擎初始化失败: {str(e)}")
            print("请检查数据文件是否存在且格式正确。")
            return False
    
    def display_welcome(self):
        """显示欢迎信息"""
        welcome_text = """
        ==================================================
                    UNESCO项目搜索引擎
        ==================================================
        欢迎使用UNESCO项目搜索引擎！
        本工具支持模糊化和口语化搜索，帮助您快速查找相关项目。
        
        命令说明:
        - 输入您的搜索查询直接搜索
        - 输入 'help' 查看详细帮助
        - 输入 'history' 查看搜索历史
        - 输入 'clear' 清空屏幕
        - 输入 'exit' 或 'quit' 退出程序
        ==================================================
        """
        print(welcome_text)
    
    def display_help(self):
        """显示帮助信息"""
        help_text = """
        UNESCO项目搜索引擎 - 帮助文档
        
        基本使用:
        1. 输入您的搜索查询(支持中文和英文)
           例如: "教育项目"、"cultural heritage in Africa"、"气候变化应对"
        
        高级选项:
        - 查询后可使用数字参数调整结果数量: "查询内容 10"(显示10条结果)
        - 支持使用方向键浏览历史查询
        
        可用命令:
        - help           : 显示此帮助信息
        - history        : 显示搜索历史
        - clear          : 清空屏幕
        - exit/quit      : 退出程序
        
        搜索技巧:
        - 使用具体术语可获得更精确的结果
        - 支持口语化表达，系统会自动转换
        - 中英文混合查询也可正常工作
        """
        print(help_text)
    
    def display_history(self):
        """显示搜索历史"""
        if not self.search_history:
            print("暂无搜索历史\n")
            return
        
        print("搜索历史:")
        for i, (query, timestamp) in enumerate(reversed(self.search_history), 1):
            time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            print(f"{i}. [{time_str}] {query}")
        print()
    
    def add_to_history(self, query: str):
        """
        添加查询到历史记录
        
        Args:
            query: 搜索查询
        """
        # 避免重复
        for existing_query, _ in self.search_history:
            if existing_query == query:
                self.search_history.remove((existing_query, _))
                break
        
        # 添加新查询
        self.search_history.append((query, datetime.now()))
        
        # 保持历史记录长度
        if len(self.search_history) > self.max_history:
            self.search_history.pop(0)
    
    def parse_input(self, user_input: str) -> tuple:
        """
        解析用户输入
        
        Args:
            user_input: 用户输入的字符串
            
        Returns:
            (查询文本, 结果数量)
        """
        parts = user_input.strip().split()
        
        # 检查最后一个部分是否为数字
        top_k = self.default_top_k
        query = user_input.strip()
        
        if parts and parts[-1].isdigit():
            try:
                num_results = int(parts[-1])
                if 1 <= num_results <= 50:  # 限制结果数量范围
                    top_k = num_results
                    # 移除数字部分
                    query = ' '.join(parts[:-1])
            except ValueError:
                pass
        
        return query, top_k
    
    def execute_search(self, query: str, top_k: int):
        """
        执行搜索并显示结果
        
        Args:
            query: 搜索查询
            top_k: 结果数量
        """
        if not query or not self.engine:
            return
        
        start_time = time.time()
        
        try:
            # 执行搜索
            results = self.engine.search(query, top_k=top_k, hybrid_weight=self.hybrid_weight)
            
            # 计算搜索时间
            search_time = time.time() - start_time
            
            # 显示结果
            print(f"搜索耗时: {search_time:.2f} 秒")
            print(self.engine.format_results(results))
            
            # 添加到历史
            self.add_to_history(query)
            
        except Exception as e:
            print(f"搜索过程中发生错误: {str(e)}")
            print("请检查您的查询或稍后再试。")
    
    def clear_screen(self):
        """清空屏幕"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def handle_command(self, command: str):
        """
        处理命令
        
        Args:
            command: 命令字符串
        """
        command = command.lower().strip()
        
        if command == 'help':
            self.display_help()
        elif command == 'history':
            self.display_history()
        elif command == 'clear':
            self.clear_screen()
            self.display_welcome()
        elif command in ['exit', 'quit', 'q']:
            print("\n感谢使用UNESCO项目搜索引擎，再见！")
            return False
        elif command:
            # 解析输入并执行搜索
            query, top_k = self.parse_input(command)
            self.execute_search(query, top_k)
        
        return True
    
    def run(self):
        """
        运行命令行界面
        """
        # 设置命令行历史
        self.setup_readline()
        
        # 显示欢迎信息
        self.display_welcome()
        
        # 初始化搜索引擎
        if not self.initialize_engine():
            print("无法初始化搜索引擎，程序将退出。")
            return
        
        try:
            # 主循环
            while True:
                try:
                    command = input("\n请输入搜索查询 (输入 'help' 获取帮助): ")
                    if not self.handle_command(command):
                        break
                except KeyboardInterrupt:
                    print("\n\n操作已取消")
                except Exception as e:
                    print(f"发生错误: {str(e)}")
        finally:
            print("\n程序已退出")

def main():
    """主函数"""
    cli = SearchCLI()
    cli.run()

if __name__ == "__main__":
    main()
