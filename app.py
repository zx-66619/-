import streamlit as st
import mysql.connector
import pandas as pd
import pickle
import numpy as np
from mysql.connector import Error
import json
import uuid
import os

# 设置页面配置
st.set_page_config(
    page_title="交通事故风险预测系统",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


class AccidentRiskApp:
    def __init__(self):
        self.db_connection = None
        self.feature_metadata = None
        self.session_id = str(uuid.uuid4())[:8]
        self.models_dir = "models"

        # 初始化session state
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'scaler' not in st.session_state:
            st.session_state.scaler = None
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'current_model' not in st.session_state:
            st.session_state.current_model = None
        if 'model_type' not in st.session_state:
            st.session_state.model_type = None

    def connect_database(self):
        """连接数据库"""
        try:
            #secrets = st.secrets["mysql"]
            self.db_connection = mysql.connector.connect(
                host='localhost',
                user='streamlit_user',
                password='123456',
                database='accident_risk_db',
                buffered=True  # 添加buffered参数避免未读结果错误
            )
            return True
        except Error as e:
            st.error(f"数据库连接失败: {e}")
            return False

    def get_available_models(self):
        """获取可用的模型列表"""
        if not os.path.exists(self.models_dir):
            return []

        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl') and 'scaler' not in f.lower()]
        return model_files

    def load_selected_model(self, model_filename):
        """加载选定的模型和相关的预处理对象"""
        try:
            # 加载模型
            model_path = os.path.join(self.models_dir, model_filename)
            with open(model_path, 'rb') as f:
                st.session_state.model = pickle.load(f)

            # 确定模型类型
            if 'linear_regression' in model_filename.lower():
                st.session_state.model_type = 'linear_regression'
            elif 'lasso' in model_filename.lower():
                st.session_state.model_type = 'lasso'
            elif 'ridge' in model_filename.lower():
                st.session_state.model_type = 'ridge'
            elif 'random_forest' in model_filename.lower():
                st.session_state.model_type = 'random_forest'
            elif 'xgboost' in model_filename.lower():
                st.session_state.model_type = 'xgboost'
            elif 'lightgbm' in model_filename.lower():
                st.session_state.model_type = 'lightgbm'
            else:
                st.session_state.model_type = 'unknown'

            # 尝试加载对应的scaler
            model_name = os.path.splitext(model_filename)[0]
            scaler_filename = f"{model_name}_scaler.pkl"
            scaler_path = os.path.join(self.models_dir, scaler_filename)

            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    st.session_state.scaler = pickle.load(f)
                return True, f"成功加载模型: {model_filename} 和特征缩放器"
            else:
                st.session_state.scaler = None
                return True, f"成功加载模型: {model_filename}，但未找到对应的特征缩放器"

        except Exception as e:
            return False, f"模型加载失败: {e}"

    def get_feature_metadata(self):
        """获取特征元数据"""
        if self.db_connection:
            try:
                cursor = self.db_connection.cursor()
                cursor.execute("SELECT * FROM feature_metadata")
                result = cursor.fetchall()
                columns = [col[0] for col in cursor.description]
                self.feature_metadata = pd.DataFrame(result, columns=columns)
                cursor.close()  # 关闭游标
                return self.feature_metadata
            except Error as e:
                st.error(f"获取特征元数据失败: {e}")
                return None

    def home_page(self):
        """主页 - 系统介绍"""
        st.title("🏠 交通事故风险预测系统")

        # 系统介绍
        st.header("系统简介")
        st.write("""
        本系统基于机器学习技术，通过对道路条件、环境因素和历史事故数据的分析，
        预测特定路段和条件下的交通事故风险等级。系统旨在帮助交通管理部门和驾驶员
        更好地了解道路安全状况，采取预防措施降低事故发生率。
        """)

        # 主要功能
        st.header("主要功能")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📊 数据可视化")
            st.write("""
            - 模型性能指标分析
            - 多种可视化图表展示
            - 交互式图表选择
            """)

        with col2:
            st.subheader("🔮 风险预测")
            st.write("""
            - 实时事故风险预测
            - 多因素综合分析
            - 风险等级评估
            """)

        with col3:
            st.subheader("📈 模型分析")
            st.write("""
            - 模型配置信息
            - 特征重要性分析
            - 学习曲线展示
            """)

        # 技术特点
        st.header("技术特点")
        st.write("""
        - **先进的机器学习算法**：采用集成学习方法，提高预测准确性
        - **全面的特征工程**：考虑道路类型、天气条件、时间因素等多维度特征
        - **实时预测能力**：基于最新数据快速评估风险等级
        - **用户友好界面**：直观的可视化展示和简洁的操作流程
        """)

        # 使用指南
        st.header("使用指南")
        with st.expander("如何开始使用系统"):
            st.write("""
            1. **数据可视化**：在左侧菜单选择"数据可视化"，查看模型性能和各种分析图表
            2. **风险预测**：选择"预测分析"，输入道路和环境参数获取风险预测
            3. **模型分析**：选择"模型分析"，深入了解模型结构和特征重要性
            """)

        # 系统统计信息
        if self.db_connection:
            try:
                cursor = self.db_connection.cursor()

                # 总记录数
                cursor.execute("SELECT COUNT(*) FROM training_data")
                total_records = cursor.fetchone()[0]

                # 模型数量
                cursor.execute("SELECT COUNT(*) FROM model_configs")
                model_count = cursor.fetchone()[0]

                # 预测记录数
                cursor.execute("SELECT COUNT(*) FROM web_predictions")
                prediction_count = cursor.fetchone()[0]

                cursor.close()  # 关闭游标

                st.header("系统统计")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("训练数据量", f"{total_records:,} 条")
                with col2:
                    st.metric("模型数量", f"{model_count} 个")
                with col3:
                    st.metric("预测次数", f"{prediction_count} 次")

            except Error as e:
                st.error(f"获取统计数据失败: {e}")

    def visualization_page(self):
        """数据可视化页面"""
        st.title("📊 数据可视化")

        st.write("选择下方图表查看数据分析结果")

        # 定义可用图表列表
        chart_options = {
            "事故概率分布直方图": "展示事故概率的整体分布情况",
            "事故概率箱线图": "显示事故概率的统计分布和异常值",
            "分类型特征事故发生率分布": "各类别特征与事故发生率的分布关系",
            "分类型各个情况事故发生平均概率": "各类别特征不同取值下的事故平均概率",
            "分类型克莱姆V热力图": "分类变量之间的关联强度热力图",
            "布尔型事故概率分布图": "布尔型特征与事故概率的分布关系",
            "布尔型事故发生平均概率分布图": "布尔型特征不同取值下的事故平均概率",
            "数值型各个情况事故分布直方图": "数值型特征与事故分布的直方图",
            "曲率——事故报告数——事故发生概率相关性热力图": "曲率、事故报告数与事故概率的相关性热力图",
            "曲率、事故报告数量分布箱线图": "曲率和事故报告数量的分布箱线图"
        }

        # 创建图表选择下拉框
        selected_chart = st.selectbox(
            "选择要查看的图表",
            options=list(chart_options.keys()),
            index=0,
            help="从下拉列表中选择一个图表进行查看"
        )

        # 显示选中的图表
        st.subheader(selected_chart)
        st.write(chart_options[selected_chart])

        # 构建图表文件路径
        chart_filename = f"{selected_chart}.png"

        try:
            # 显示图表
            st.image(chart_filename, use_column_width=True)
            st.success(f"成功加载图表: {chart_filename}")
        except Exception as e:
            st.error(f"无法加载图表: {chart_filename}")
            st.info(f"请确保文件 '{chart_filename}' 存在于当前目录中")

    def prediction_page(self):
        """预测分析页面"""
        st.title("🔮 事故风险预测")

        st.write("使用训练好的模型对新数据进行事故风险预测")

        # 获取可用的模型列表
        available_models = self.get_available_models()

        if not available_models:
            st.error("未找到任何模型文件。请确保models文件夹中存在.pkl格式的模型文件。")
            st.info("模型文件应该放在 'models' 文件夹中")
            return

        # 模型选择部分
        st.header("1. 选择预测模型")

        # 设置默认模型为 lightgbm
        default_index = 0
        for i, model in enumerate(available_models):
            if 'lightgbm' in model.lower():
                default_index = i
                break
            elif 'xgboost' in model.lower():
                default_index = i  # 如果没有lightgbm，使用xgboost作为备选

        selected_model = st.selectbox(
            "选择要使用的预测模型",
            options=available_models,
            index=default_index,
            help="从下拉列表中选择一个模型进行预测"
        )

        # 检查是否需要重新加载模型
        need_reload = (not st.session_state.model_loaded or
                       st.session_state.current_model != selected_model)

        if need_reload:
            st.session_state.model_loaded = False

        # 加载模型按钮
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("加载模型", type="primary", key="load_model_btn"):
                with st.spinner(f"正在加载模型 {selected_model}..."):
                    success, message = self.load_selected_model(selected_model)
                    if success:
                        st.session_state.model_loaded = True
                        st.session_state.current_model = selected_model
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)

        with col2:
            if st.session_state.model_loaded:
                st.success(f"✅ 模型已加载: {st.session_state.current_model}")
                st.info(f"模型类型: {st.session_state.model_type}")
                if st.session_state.scaler:
                    st.info("✅ 特征缩放器已加载")
                else:
                    st.warning("⚠️ 未找到特征缩放器")
            else:
                st.warning("⚠️ 请先加载模型")

        # 如果模型未加载，显示提示并返回
        if not st.session_state.model_loaded:
            st.info("请先点击'加载模型'按钮加载选定的模型")
            return

        # 创建预测表单
        st.header("2. 输入预测参数")

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            input_features = {}

            with col1:
                # 道路类型
                road_type = st.selectbox(
                    "道路类型",
                    options=['urban', 'rural', 'highway'],
                    index=0,
                    help="选择道路类型"
                )
                input_features['road_type'] = road_type

                # 车道数量
                num_lanes = st.slider(
                    "车道数量",
                    min_value=1,
                    max_value=8,
                    value=2,
                    help="选择车道数量"
                )
                input_features['num_lanes'] = num_lanes

                # 道路曲率
                curvature = st.slider(
                    "道路曲率",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    help="道路曲率，0表示直线，1表示急弯"
                )
                input_features['curvature'] = curvature

                # 限速
                speed_limit = st.slider(
                    "限速 (km/h)",
                    min_value=20,
                    max_value=120,
                    value=60,
                    help="道路限速"
                )
                input_features['speed_limit'] = speed_limit

                # 光照条件
                lighting = st.selectbox(
                    "光照条件",
                    options=['daylight', 'dim', 'night'],
                    index=0,
                    help="选择光照条件"
                )
                input_features['lighting'] = lighting

            with col2:
                # 天气状况
                weather = st.selectbox(
                    "天气状况",
                    options=['clear', 'rainy', 'foggy'],
                    index=0,
                    help="选择天气状况"
                )
                input_features['weather'] = weather

                # 道路标志
                road_signs_present = st.checkbox(
                    "是否有道路标志",
                    value=True,
                    help="道路是否有交通标志"
                )
                input_features['road_signs_present'] = road_signs_present

                # 公共道路
                public_road = st.checkbox(
                    "是否公共道路",
                    value=True,
                    help="是否为公共道路"
                )
                input_features['public_road'] = public_road

                # 时间段
                time_of_day = st.selectbox(
                    "时间段",
                    options=['morning', 'afternoon', 'evening'],
                    index=1,
                    help="选择时间段"
                )
                input_features['time_of_day'] = time_of_day

                # 节假日
                holiday = st.checkbox(
                    "是否节假日",
                    value=False,
                    help="是否为节假日"
                )
                input_features['holiday'] = holiday

                # 学校季节
                school_season = st.checkbox(
                    "是否学校季节",
                    value=False,
                    help="是否为学校开学季节"
                )
                input_features['school_season'] = school_season

                # 报告事故数量
                num_reported_accidents = st.slider(
                    "报告事故数量",
                    min_value=0,
                    max_value=10,
                    value=1,
                    help="历史报告事故数量"
                )
                input_features['num_reported_accidents'] = num_reported_accidents

            # 提交按钮
            submitted = st.form_submit_button("进行风险预测", type="primary")

            if submitted:
                self.make_prediction(input_features)

    def create_features_for_model(self, input_features, model_type):
        """根据模型类型创建对应的特征"""
        # 基础特征
        road_type = input_features['road_type']
        num_lanes = input_features['num_lanes']
        curvature = input_features['curvature']
        speed_limit = input_features['speed_limit']
        lighting = input_features['lighting']
        weather = input_features['weather']
        road_signs_present = input_features['road_signs_present']
        public_road = input_features['public_road']
        time_of_day = input_features['time_of_day']
        holiday = input_features['holiday']
        school_season = input_features['school_season']
        num_reported_accidents = input_features['num_reported_accidents']

        # 创建特征字典
        features = {}

        if model_type == 'linear_regression':
            # 线性回归特征
            features['num_reported_accidents_log_scaled'] = np.log1p(num_reported_accidents)
            features['num_lanes_enc_scaled'] = num_lanes / 8.0
            features['speed_limit_enc_scaled'] = speed_limit / 120.0
            features['holiday'] = 1 if holiday else 0
            features['public_road'] = 1 if public_road else 0
            features['road_signs_present'] = 1 if road_signs_present else 0
            features['school_season'] = 1 if school_season else 0

            # One-hot编码特征
            features['road_type_highway'] = 1 if road_type == 'highway' else 0
            features['road_type_rural'] = 1 if road_type == 'rural' else 0
            features['road_type_urban'] = 1 if road_type == 'urban' else 0

            features['weather_clear'] = 1 if weather == 'clear' else 0
            features['weather_foggy'] = 1 if weather == 'foggy' else 0
            features['weather_rainy'] = 1 if weather == 'rainy' else 0

            features['time_of_day_afternoon'] = 1 if time_of_day == 'afternoon' else 0
            features['time_of_day_evening'] = 1 if time_of_day == 'evening' else 0
            features['time_of_day_morning'] = 1 if time_of_day == 'morning' else 0

            # 交互特征
            features['curvature_speed_scaled'] = curvature * (speed_limit / 120.0)
            features['curvature_night_scaled'] = curvature * (1 if lighting == 'night' else 0)

        elif model_type in ['lasso', 'ridge']:
            # Lasso和Ridge回归特征 - 只使用训练时使用的特征
            features['num_reported_accidents_log_scaled'] = np.log1p(num_reported_accidents)
            features['num_lanes_enc_scaled'] = num_lanes / 8.0
            features['speed_limit_enc_scaled'] = speed_limit / 120.0
            features['public_road'] = 1 if public_road else 0
            features['road_signs_present'] = 1 if road_signs_present else 0
            features['weather_clear'] = 1 if weather == 'clear' else 0
            features['weather_rainy'] = 1 if weather == 'rainy' else 0
            features['time_of_day_evening'] = 1 if time_of_day == 'evening' else 0
            features['curvature_speed_scaled'] = curvature * (speed_limit / 120.0)
            features['curvature_night_scaled'] = curvature * (1 if lighting == 'night' else 0)

        elif model_type == 'random_forest':
            # 随机森林特征
            features['curvature_speed'] = curvature * speed_limit
            features['curvature_night'] = curvature * (1 if lighting == 'night' else 0)
            features['speed_limit_enc'] = speed_limit / 120.0
            features['curvature'] = curvature
            features['weather_clear'] = 1 if weather == 'clear' else 0
            features['lighting_night'] = 1 if lighting == 'night' else 0
            features['num_reported_accidents'] = num_reported_accidents

        elif model_type == 'xgboost':
            # XGBoost特征 - 根据错误信息，训练时只使用了7个特征
            features['curvature_speed'] = float(curvature * speed_limit)
            features['curvature_night'] = float(curvature * (1 if lighting == 'night' else 0))

            # 对于分类特征，使用整数编码而不是浮点数
            lighting_map = {'daylight': 0, 'dim': 1, 'night': 2}
            weather_map = {'clear': 0, 'rainy': 1, 'foggy': 2}

            features['lighting'] = lighting_map[lighting]
            features['speed_limit_enc'] = float(speed_limit / 120.0)
            features['weather'] = weather_map[weather]
            features['curvature'] = float(curvature)
            features['num_reported_accidents'] = float(num_reported_accidents)

        elif model_type == 'lightgbm':
            # LightGBM特征 - 确保分类特征正确设置
            features['curvature'] = curvature
            features['curvature_speed'] = curvature * speed_limit
            features['weather'] = {'clear': 0, 'rainy': 1, 'foggy': 2}[weather]
            features['speed_limit'] = speed_limit
            features['num_reported_accidents'] = num_reported_accidents
            features['curvature_night'] = curvature * (1 if lighting == 'night' else 0)
            features['lighting'] = {'daylight': 0, 'dim': 1, 'night': 2}[lighting]
            features['public_road'] = 1 if public_road else 0
            features['holiday'] = 1 if holiday else 0
            features['num_lanes'] = num_lanes
            features['time_of_day'] = {'morning': 0, 'afternoon': 1, 'evening': 2}[time_of_day]
            features['road_type'] = {'urban': 0, 'rural': 1, 'highway': 2}[road_type]
            features['road_signs_present'] = 1 if road_signs_present else 0
            features['school_season'] = 1 if school_season else 0

        else:
            # 未知模型类型，使用基础特征
            features = input_features.copy()
            # 将布尔值转换为0/1
            for key in features:
                if isinstance(features[key], bool):
                    features[key] = 1 if features[key] else 0

        return features

    def preprocess_features(self, input_features):
        """预处理输入特征，转换为模型需要的格式"""
        # 根据模型类型创建特征
        model_type = st.session_state.model_type
        features_dict = self.create_features_for_model(input_features, model_type)

        # 创建DataFrame
        features_df = pd.DataFrame([features_dict])

        # 对于XGBoost模型，确保特征顺序与训练时一致
        if model_type == 'xgboost':
            # 根据错误信息，训练时使用的特征顺序
            expected_features_order = [
                'curvature_speed', 'curvature_night', 'lighting', 'speed_limit_enc',
                'weather', 'curvature', 'num_reported_accidents'
            ]
            # 只保留训练时使用的特征，并按正确顺序排列
            features_df = features_df[expected_features_order]

        # 对于Lasso和Ridge模型，简化特征缩放处理
        if model_type in ['lasso', 'ridge'] and st.session_state.scaler:
            try:
                # 只对数值特征进行缩放，忽略特征名称
                numerical_features = features_df.select_dtypes(include=[np.number]).columns
                features_df[numerical_features] = st.session_state.scaler.transform(features_df[numerical_features])
            except Exception:
                # 如果缩放失败，继续使用原始特征进行预测
                pass

        # 对于其他模型，如果有scaler，直接应用
        elif st.session_state.scaler:
            try:
                features_df = pd.DataFrame(
                    st.session_state.scaler.transform(features_df),
                    columns=features_df.columns
                )
            except Exception:
                # 如果缩放失败，继续使用原始特征进行预测
                pass

        # 对于LightGBM，设置分类特征
        if model_type == 'lightgbm':
            categorical_features = ['weather', 'lighting', 'time_of_day', 'road_type']
            for feature in categorical_features:
                if feature in features_df.columns:
                    features_df[feature] = features_df[feature].astype('category')

        # 对于XGBoost，确保所有特征都是数值型，并且使用正确的数据类型
        if model_type == 'xgboost':
            # 确保所有特征都是数值型
            for col in features_df.columns:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')

            # 填充可能的NaN值
            features_df = features_df.fillna(0)

            # 确保数据类型一致
            features_df = features_df.astype(np.float32)

        return features_df

    def make_prediction(self, input_features):
        """进行预测"""
        try:
            # 检查模型是否已加载
            if st.session_state.model is None:
                st.error("模型未加载，请先加载模型")
                return

            # 预处理特征
            features_processed = self.preprocess_features(input_features)

            if features_processed is None:
                st.error("特征预处理失败，无法进行预测")
                return

            # 检查特征数量
            expected_features_count = {
                'linear_regression': 18,
                'lasso': 10,
                'ridge': 10,
                'random_forest': 7,
                'xgboost': 7,  # XGBoost现在只使用7个特征
                'lightgbm': 14
            }

            model_type = st.session_state.model_type
            if model_type in expected_features_count:
                expected_count = expected_features_count[model_type]
                actual_count = len(features_processed.columns)
                if actual_count != expected_count:
                    st.warning(f"特征数量: 期望 {expected_count} 个，实际 {actual_count} 个")

            # 进行预测
            if model_type == 'xgboost':
                # 对于XGBoost，确保使用正确的预测方法
                try:
                    # 尝试直接预测
                    prediction = st.session_state.model.predict(features_processed)[0]
                except Exception as e:
                    st.error(f"XGBoost预测失败: {e}")
                    # 尝试使用predict_proba（如果是分类问题）
                    try:
                        prediction_proba = st.session_state.model.predict_proba(features_processed)
                        prediction = prediction_proba[0][1] if prediction_proba.shape[1] > 1 else prediction_proba[0][0]
                    except:
                        # 最后尝试使用原始预测值
                        prediction = st.session_state.model.predict(features_processed, output_margin=True)[0]
                        # 如果是margin输出，使用sigmoid转换
                        prediction = 1 / (1 + np.exp(-prediction))
            else:
                prediction = st.session_state.model.predict(features_processed)[0]

            # 确保预测值在合理范围内
            prediction = max(0.0, min(1.0, float(prediction)))

            # 确定风险等级
            if prediction < 0.3:
                risk_level = 'low'
            elif prediction < 0.7:
                risk_level = 'medium'
            else:
                risk_level = 'high'

            # 显示预测结果
            st.header("📊 预测结果")

            # 使用columns布局显示结果
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                # 风险值显示
                st.metric("预测风险值", f"{prediction:.4f}")

            with col2:
                # 风险等级显示
                st.metric("风险等级", risk_level.upper())

            with col3:
                # 风险颜色指示
                if risk_level == 'low':
                    st.success("低风险")
                elif risk_level == 'medium':
                    st.warning("中等风险")
                else:
                    st.error("高风险")

            # 风险可视化进度条
            st.subheader("风险程度可视化")
            st.progress(float(prediction), text=f"风险程度: {prediction:.2%}")

            # 风险解释
            st.subheader("风险说明")
            if risk_level == 'low':
                st.info("""
                **低风险说明**: 当前条件下事故风险较低，但仍需保持谨慎驾驶。
                - 建议：保持正常驾驶习惯，注意观察路况
                """)
            elif risk_level == 'medium':
                st.warning("""
                **中等风险说明**: 当前条件下存在一定事故风险，需要提高警惕。
                - 建议：减速慢行，加强观察，保持安全车距
                """)
            else:
                st.error("""
                **高风险说明**: 当前条件下事故风险较高，需要特别小心。
                - 建议：显著降低车速，保持高度警惕，必要时选择其他路线
                """)

            # 保存预测记录到数据库 - 修复后的代码
            if self.db_connection:
                try:
                    # 使用一个游标查询模型ID
                    cursor1 = self.db_connection.cursor()
                    cursor1.execute("SELECT id FROM model_configs WHERE is_active = TRUE")
                    result = cursor1.fetchone()
                    model_config_id = result[0] if result else 1
                    cursor1.close()  # 关闭查询游标

                    # 使用另一个游标执行插入
                    cursor2 = self.db_connection.cursor()
                    cursor2.execute("""
                       INSERT INTO web_predictions
                       (model_config_id, input_features, predicted_risk, risk_level, session_id)
                       VALUES (%s, %s, %s, %s, %s)
                       """, (model_config_id, json.dumps(input_features), float(prediction), risk_level,
                             self.session_id))

                    self.db_connection.commit()
                    cursor2.close()  # 关闭插入游标
                    st.success("✅ 预测完成！预测记录已保存到数据库。")

                except Error as e:
                    st.warning(f"⚠️ 预测记录保存失败: {e}，但预测已完成")

        except Exception as e:
            st.error(f"❌ 预测过程中出现错误: {e}")

            # 提供详细的调试信息
            with st.expander("调试信息"):
                st.write(f"当前模型: {st.session_state.current_model}")
                st.write(f"模型类型: {st.session_state.model_type}")
                if 'features_processed' in locals():
                    st.write(f"实际特征: {features_processed.columns.tolist()}")
                    st.write(f"实际特征数量: {len(features_processed.columns)}")
                    st.write(f"特征值: {features_processed.iloc[0].to_dict()}")
                    st.write(f"特征数据类型:")
                    for col in features_processed.columns:
                        st.write(f"- {col}: {features_processed[col].dtype}")

    def model_analysis_page(self):
        """模型分析页面"""
        st.title("📈 模型分析")

        # 学习曲线
        st.header("学习曲线")

        # 尝试加载当前目录下的学习曲线图片
        learning_curve_files = [
            "学习曲线.png",
            "learning_curve.png",
            "learning_curves.png"
        ]

        learning_curve_loaded = False
        for curve_file in learning_curve_files:
            if os.path.exists(curve_file):
                try:
                    st.image(curve_file, use_column_width=True, caption="模型学习曲线")
                    st.success(f"成功加载学习曲线: {curve_file}")
                    learning_curve_loaded = True
                    break
                except Exception as e:
                    continue

        if not learning_curve_loaded:
            st.warning("无法找到学习曲线图片文件。请确保以下文件之一存在于当前目录:")
            for curve_file in learning_curve_files:
                st.write(f"- {curve_file}")

        # 模型特征重要性分析
        st.header("模型特征重要性分析")

        # 定义可用的模型列表
        model_options = ["LightGBM", "XGBoost", "RandomForest", "Lasso", "Ridge"]

        # 创建下拉选择框
        selected_model = st.selectbox(
            "选择要分析的模型",
            options=model_options,
            index=0,
            help="选择模型查看其特征重要性分析"
        )

        # 构建对应的图片文件名
        feature_importance_file = f"{selected_model}模型特征重要性.png"

        # 尝试加载特征重要性图片
        try:
            if os.path.exists(feature_importance_file):
                st.image(feature_importance_file, use_column_width=True, caption=f"{selected_model}模型特征重要性分析")
                st.success(f"成功加载特征重要性分析图: {feature_importance_file}")
            else:
                st.warning(f"未找到特征重要性分析图片: {feature_importance_file}")
                st.info(f"请确保文件 '{feature_importance_file}' 存在于当前目录中")
        except Exception as e:
            st.error(f"加载特征重要性分析图失败: {e}")

        # 模型性能对比图
        st.header("模型性能对比图")

        # 尝试加载模型性能对比图
        performance_comparison_files = [
            "模型性能对比图.png",
            "model_performance_comparison.png",
            "performance_comparison.png"
        ]

        performance_comparison_loaded = False
        for perf_file in performance_comparison_files:
            if os.path.exists(perf_file):
                try:
                    st.image(perf_file, use_column_width=True, caption="模型性能对比图")
                    st.success(f"成功加载模型性能对比图: {perf_file}")
                    performance_comparison_loaded = True
                    break
                except Exception as e:
                    continue

        if not performance_comparison_loaded:
            st.warning("无法找到模型性能对比图图片文件。请确保以下文件之一存在于当前目录:")
            for perf_file in performance_comparison_files:
                st.write(f"- {perf_file}")

        # 残差分析图
        st.header("残差分析图")

        # 尝试加载残差分析图
        residual_analysis_files = [
            "残差分析图.png",
            "residual_analysis.png",
            "residuals_plot.png"
        ]

        residual_analysis_loaded = False
        for residual_file in residual_analysis_files:
            if os.path.exists(residual_file):
                try:
                    st.image(residual_file, use_column_width=True, caption="残差分析图")
                    st.success(f"成功加载残差分析图: {residual_file}")
                    residual_analysis_loaded = True
                    break
                except Exception as e:
                    continue

        if not residual_analysis_loaded:
            st.warning("无法找到残差分析图图片文件。请确保以下文件之一存在于当前目录:")
            for residual_file in residual_analysis_files:
                st.write(f"- {residual_file}")

        # 模型性能指标
        st.header("模型性能指标")

        if self.db_connection:
            try:
                cursor = self.db_connection.cursor()

                # 获取模型性能数据
                cursor.execute("""
                               SELECT mc.model_name, mp.dataset_type, mp.mse, mp.r2_score, mp.mae
                               FROM model_performance mp
                                        JOIN model_configs mc ON mp.model_config_id = mc.id
                               WHERE mc.is_active = TRUE
                               """)
                performance_data = cursor.fetchall()

                if performance_data:
                    # 创建性能指标表格 - 移除了RMSE列
                    perf_df = pd.DataFrame(performance_data,
                                           columns=['模型名称', '数据集', 'MSE', 'R2 Score', 'MAE'])
                    st.dataframe(perf_df.style.format({
                        'MSE': '{:.4f}',
                        'R2 Score': '{:.4f}',
                        'MAE': '{:.4f}'
                    }))

                    # 添加性能指标解释 - 更新说明，移除了RMSE
                    with st.expander("性能指标说明"):
                        st.write("""
                        - **MSE (均方误差)**: 预测值与真实值之差的平方的平均值，值越小越好
                        - **R2 Score (决定系数)**: 表示模型解释的方差比例，值越接近1越好
                        - **MAE (平均绝对误差)**: 预测值与真实值之差的绝对值的平均值，值越小越好
                        """)
                else:
                    st.info("暂无模型性能数据")

                cursor.close()  # 关闭游标

            except Error as e:
                st.error(f"加载模型性能数据失败: {e}")
        else:
            st.error("数据库连接失败，无法加载模型性能指标")

    def run(self):
        """运行应用"""
        # 初始化连接
        if not self.db_connection:
            if not self.connect_database():
                st.error("无法连接到数据库，请检查数据库配置")
                return

        # 加载特征元数据
        self.get_feature_metadata()

        # 侧边栏导航
        st.sidebar.title("🚗 导航菜单")

        # 在侧边栏添加logo或标题
        st.sidebar.markdown("---")

        # 导航选项 - 现在有四个选项
        page = st.sidebar.radio(
            "选择功能模块",
            ["主页", "数据可视化", "预测分析", "模型分析"]
        )

        # 在侧边栏添加模型状态信息
        st.sidebar.markdown("---")
        st.sidebar.subheader("系统状态")

        # 显示数据库连接状态
        db_status = "✅ 已连接" if self.db_connection and self.db_connection.is_connected() else "❌ 未连接"
        st.sidebar.write(f"数据库: {db_status}")

        # 显示模型加载状态
        model_status = "✅ 已加载" if st.session_state.model_loaded else "❌ 未加载"
        st.sidebar.write(f"预测模型: {model_status}")

        if st.session_state.model_loaded:
            st.sidebar.write(f"当前模型: {st.session_state.current_model}")
            st.sidebar.write(f"模型类型: {st.session_state.model_type}")

        st.sidebar.markdown("---")
        st.sidebar.info("交通事故风险预测系统 v1.0")

        # 根据选择显示对应页面s
        if page == "主页":
            self.home_page()
        elif page == "数据可视化":
            self.visualization_page()
        elif page == "预测分析":
            self.prediction_page()
        elif page == "模型分析":
            self.model_analysis_page()


# 运行应用
if __name__ == "__main__":
    app = AccidentRiskApp()
    app.run()