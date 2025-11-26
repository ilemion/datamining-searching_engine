import pandas as pd
import re
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# 区域到坐标的映射字典（常见区域的大致中心坐标）
REGION_COORDINATES = {
    'Africa': 'Africa',
    'Asia and the Pacific': 'Asia-Pacific',
    'Europe and North America': 'Europe',
    'Latin America and the Caribbean': 'Latin America',
    'Arab States': 'Middle East',
    'Global': None  # 全球性项目不指定特定坐标
}

def extract_coordinates_from_text(text):
    """从文本中提取可能的地理位置信息"""
    if not text or pd.isna(text):
        return None
    
    # 尝试直接提取坐标格式 (latitude, longitude)
    coord_pattern = r'\b(\d{1,3}\.?\d*)\s*[,;]\s*(-?\d{1,3}\.?\d*)\b'
    match = re.search(coord_pattern, str(text))
    if match:
        try:
            lat, lon = float(match.group(1)), float(match.group(2))
            # 简单验证坐标范围
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return f"{lat}, {lon}"
        except:
            pass
    
    # 尝试识别国家或地区名称
    country_keywords = [
        'China', 'India', 'United States', 'Brazil', 'Australia', 'Russia',
        'France', 'Germany', 'Japan', 'South Africa', 'Nigeria', 'Egypt',
        'Kenya', 'Morocco', 'Tanzania', 'Ghana', 'Ethiopia', 'Uganda',
        'Indonesia', 'Thailand', 'Vietnam', 'Philippines', 'Malaysia',
        'Singapore', 'Myanmar', 'Cambodia', 'Lao PDR', 'Bangladesh',
        'Pakistan', 'Afghanistan', 'Iran', 'Iraq', 'Saudi Arabia',
        'United Arab Emirates', 'Israel', 'Jordan', 'Lebanon', 'Syria',
        'Turkey', 'Greece', 'Italy', 'Spain', 'Portugal', 'United Kingdom',
        'Ireland', 'Netherlands', 'Belgium', 'Switzerland', 'Austria',
        'Poland', 'Ukraine', 'Romania', 'Bulgaria', 'Hungary',
        'Czech Republic', 'Slovakia', 'Slovenia', 'Croatia', 'Bosnia and Herzegovina',
        'Serbia', 'Montenegro', 'North Macedonia', 'Albania', 'Kosovo',
        'Moldova', 'Belarus', 'Estonia', 'Latvia', 'Lithuania',
        'Finland', 'Sweden', 'Norway', 'Denmark', 'Iceland',
        'Mexico', 'Guatemala', 'Honduras', 'El Salvador', 'Nicaragua',
        'Costa Rica', 'Panama', 'Colombia', 'Venezuela', 'Ecuador',
        'Peru', 'Bolivia', 'Chile', 'Argentina', 'Uruguay', 'Paraguay',
        'Canada', 'New Zealand', 'South Korea', 'North Korea', 'Mongolia',
        'Taiwan', 'Hong Kong', 'Macau', 'Bahrain', 'Kuwait', 'Qatar',
        'Oman', 'Yemen', 'Palestine', 'Cyprus', 'Armenia', 'Azerbaijan',
        'Georgia', 'Kazakhstan', 'Uzbekistan', 'Turkmenistan', 'Kyrgyzstan',
        'Tajikistan', 'Nepal', 'Bhutan', 'Sri Lanka', 'Maldives',
        'Timor-Leste', 'Brunei', 'Laos', 'Cuba', 'Haiti', 'Dominican Republic',
        'Jamaica', 'Trinidad and Tobago', 'Barbados', 'Grenada', 'St. Lucia',
        'Antigua and Barbuda', 'Dominica', 'St. Vincent and the Grenadines',
        'Belize', 'Guyana', 'Suriname', 'French Guiana', 'Bahamas', 'Curaçao',
        'Aruba', 'Sint Maarten', 'Saint Kitts and Nevis', 'Saint Martin',
        'Anguilla', 'Montserrat', 'British Virgin Islands', 'US Virgin Islands',
        'Turks and Caicos Islands', 'Cayman Islands', 'Puerto Rico'
    ]
    
    # 转换文本为小写以便匹配
    text_lower = str(text).lower()
    
    # 检查是否包含国家名称
    for country in country_keywords:
        if country.lower() in text_lower:
            return country  # 返回国家名称，后续可以通过地理编码获取坐标
    
    return None

def complete_coordinates(input_file, output_file):
    """补全CSV文件中的coordinates字段"""
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 统计需要补全的记录数
    total_records = len(df)
    empty_coordinates = df['Coordinates'].isna() | (df['Coordinates'] == '')
    records_to_complete = empty_coordinates.sum()
    
    print(f"总记录数: {total_records}")
    print(f"需要补全coordinates的记录数: {records_to_complete}")
    
    # 创建地理编码器（用于后续可能的地理编码，这里先不实际调用以避免API限制）
    # geolocator = Nominatim(user_agent="unesco_coordinates_completion")
    # geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    
    # 记录补全统计
    completed_from_regional = 0
    completed_from_description = 0
    still_missing = 0
    
    # 处理每一行
    for index, row in df.iterrows():
        # 如果coordinates已经有值，跳过
        if not pd.isna(row['Coordinates']) and row['Coordinates'] != '':
            continue
        
        # 尝试从Regional Group补全
        if not pd.isna(row['Regional Group']) and row['Regional Group'] != '':
            regional_group = row['Regional Group']
            for region_key, region_value in REGION_COORDINATES.items():
                if region_key in regional_group:
                    df.at[index, 'Coordinates'] = region_value if region_value else None
                    completed_from_regional += 1
                    break
            else:
                # 如果没有匹配到预定义的区域，尝试提取其他地理信息
                extracted = extract_coordinates_from_text(regional_group)
                if extracted:
                    df.at[index, 'Coordinates'] = extracted
                    completed_from_regional += 1
        
        # 如果从Regional Group没有补全，尝试从Description补全
        if pd.isna(df.at[index, 'Coordinates']) or df.at[index, 'Coordinates'] == '':
            extracted = extract_coordinates_from_text(row['Description'])
            if extracted:
                df.at[index, 'Coordinates'] = extracted
                completed_from_description += 1
            else:
                # 都没有则设置为Global
                df.at[index, 'Coordinates'] = 'Global'
                still_missing += 1
    
    # 统计结果
    print(f"从Regional Group补全的记录数: {completed_from_regional}")
    print(f"从Description补全的记录数: {completed_from_description}")
    print(f"仍然缺失的记录数: {still_missing}")
    
    # 保存补全后的文件
    df.to_csv(output_file, index=False)
    print(f"补全后的文件已保存到: {output_file}")
    
    # 显示前10行的示例
    print("\n补全后的前10行示例:")
    print(df[['Project', 'Regional Group', 'Description', 'Coordinates']].head(10))

if __name__ == "__main__":
    input_file = "UNESCO_projects_filtered_searching_engine.csv"
    output_file = "UNESCO_projects_coordinates_completed.csv"
    
    try:
        complete_coordinates(input_file, output_file)
    except Exception as e:
        print(f"处理过程中出错: {e}")
