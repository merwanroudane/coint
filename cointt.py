import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, coint
from statsmodels.tsa.api import VAR
import io
from PIL import Image
import base64
import urllib.request
from io import BytesIO

# تعيين عنوان وشكل التطبيق
st.set_page_config(
	page_title="التحليل الاقتصادي القياسي: VAR, VECM",
	page_icon="📊",
	layout="wide",
)

# إضافة CSS للتصميم
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 30px;
    }
    .sub-header {
        font-size: 26px;
        font-weight: bold;
        color: #0D47A1;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    .concept {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .formula-box {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 20px;
    }
    .note {
        background-color: #fff8e1;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #FFC107;
        margin-bottom: 20px;
    }
    .application {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin-bottom: 20px;
    }
    .caution {
        background-color: #ffebee;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #f44336;
        margin-bottom: 20px;
    }
    .conclusion {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        margin-top: 30px;
        margin-bottom: 30px;
    }
    .rtl {
        direction: rtl;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)


# وظيفة لإنشاء بيانات المثال
def generate_sample_data(n=200, coef=0.5):
	np.random.seed(42)
	# إنشاء سلاسل زمنية متكاملة من الدرجة الأولى I(1)
	e1 = np.random.normal(0, 1, n)
	e2 = np.random.normal(0, 1, n)

	# سلسلة زمنية أولى
	y1 = np.zeros(n)
	y1[0] = e1[0]
	for t in range(1, n):
		y1[t] = y1[t - 1] + e1[t]

	# سلسلة زمنية ثانية مرتبطة بالأولى لإنشاء علاقة تكامل مشترك
	y2 = np.zeros(n)
	y2 = coef * y1 + e2

	# سلسلة ثالثة مستقلة للمقارنة
	y3 = np.zeros(n)
	y3[0] = np.random.normal(0, 1)
	for t in range(1, n):
		y3[t] = y3[t - 1] + np.random.normal(0, 1)

	# تحويل إلى DataFrame
	data = pd.DataFrame({
		'y1': y1,
		'y2': y2,
		'y3': y3
	})

	return data


# وظيفة لرسم السلاسل الزمنية
def plot_time_series(data):
	fig = go.Figure()
	for col in data.columns:
		fig.add_trace(go.Scatter(x=data.index, y=data[col], mode='lines', name=col))
	fig.update_layout(
		title="السلاسل الزمنية",
		xaxis_title="الزمن",
		yaxis_title="القيمة",
		height=500,
		template="plotly_white",
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
	)
	return fig


# وظيفة لإجراء اختبار جذر الوحدة (Augmented Dickey-Fuller)
def run_adf_test(series):
	result = adfuller(series)
	return {
		'Test Statistic': result[0],
		'p-value': result[1],
		'Critical Values': result[4],
		'Stationary': result[1] < 0.05
	}


# وظيفة لإجراء اختبار التكامل المشترك بطريقة انجل-جرانجر
def run_engle_granger(y1, y2):
	# الانحدار
	X = y1.values.reshape(-1, 1)
	result = np.linalg.lstsq(X, y2, rcond=None)[0]

	# حساب البواقي
	residuals = y2 - result[0] * y1

	# اختبار ADF على البواقي
	adf_result = adfuller(residuals)

	return {
		'Coefficient': result[0],
		'ADF Test Statistic': adf_result[0],
		'ADF p-value': adf_result[1],
		'ADF Critical Values': adf_result[4],
		'Cointegration': adf_result[1] < 0.05,
		'Residuals': residuals
	}


# وظيفة لإجراء اختبار التكامل المشترك بطريقة جوهانسون
def run_johansen(data):
	try:
		result = coint_johansen(data, det_order=0, k_ar_diff=1)
		trace_stats = result.lr1
		max_eig_stats = result.lr2
		trace_crit = result.cvt
		max_eig_crit = result.cvm

		rank_trace = sum(trace_stats > trace_crit[:, 0])
		rank_max_eig = sum(max_eig_stats > max_eig_crit[:, 0])

		return {
			'Trace Statistics': trace_stats,
			'Max Eigenvalue Statistics': max_eig_stats,
			'Trace Critical Values (95%)': trace_crit[:, 0],
			'Max Eigenvalue Critical Values (95%)': max_eig_crit[:, 0],
			'Cointegration Rank (Trace)': rank_trace,
			'Cointegration Rank (Max Eigenvalue)': rank_max_eig
		}
	except:
		return "غير قابل للتطبيق - تأكد من وجود متغيرين مستقرين على الأقل"


# وظيفة لتقدير نموذج VAR
def run_var_model(data, maxlags=10):
	model = VAR(data)
	results = {}

	# تحديد عدد التأخيرات المثلى
	results['lag_order'] = model.select_order(maxlags=maxlags)

	# اختيار التأخير الأمثل باستخدام معيار AIC
	p = results['lag_order'].aic

	# تقدير النموذج
	var_model = model.fit(p)
	results['model'] = var_model
	results['summary'] = var_model.summary()

	# التنبؤ
	results['forecast'] = var_model.forecast(data.values[-p:], 10)

	# دالة الاستجابة النبضية
	results['irf'] = var_model.irf(10)

	# تحليل تجزئة التباين
	results['fevd'] = var_model.fevd(10)

	return results


# وظيفة لتقدير نموذج VECM
def run_vecm_model(data, k_ar_diff=1, coint_rank=1):
	try:
		model = VECM(data, k_ar_diff=k_ar_diff, coint_rank=coint_rank, deterministic="ci")
		results = model.fit()
		return {
			'model': results,
			'summary': results.summary(),
			'alpha': results.alpha,
			'beta': results.beta
		}
	except:
		return "غير قابل للتطبيق - تأكد من تحديد رتبة التكامل المشترك بشكل صحيح"


# وظيفة لاختبار السببية لجرانجر
def run_granger_causality(data, maxlag=5):
	results = {}
	for i in range(len(data.columns)):
		for j in range(len(data.columns)):
			if i != j:
				try:
					result = grangercausalitytests(data[[data.columns[j], data.columns[i]]], maxlag=maxlag,
												   verbose=False)
					min_p_value = min([result[lag + 1][0]['ssr_ftest'][1] for lag in range(maxlag)])
					results[f"{data.columns[i]} تؤثر على {data.columns[j]}"] = {
						'Min p-value': min_p_value,
						'Significant': min_p_value < 0.05
					}
				except:
					results[f"{data.columns[i]} تؤثر على {data.columns[j]}"] = "غير قابل للتطبيق"

	return results


# وظيفة لرسم دالة الاستجابة النبضية
def plot_irf(irf_results, var_names):
	figs = []
	for i, name in enumerate(var_names):
		fig = go.Figure()
		for j, response_name in enumerate(var_names):
			fig.add_trace(go.Scatter(
				x=np.arange(len(irf_results.irfs[:, j, i])),
				y=irf_results.irfs[:, j, i],
				mode='lines',
				name=f'استجابة {response_name}'
			))
		fig.update_layout(
			title=f'دالة الاستجابة النبضية: صدمة في {name}',
			xaxis_title="الفترات",
			yaxis_title="الاستجابة",
			height=500,
			template="plotly_white"
		)
		figs.append(fig)
	return figs


# وظيفة لرسم تجزئة التباين
def plot_fevd(fevd_results, var_names):
	figs = []
	for i, name in enumerate(var_names):
		fig = go.Figure()
		fevd_data = fevd_results.decomp[i]
		for j, source_name in enumerate(var_names):
			fig.add_trace(go.Scatter(
				x=np.arange(fevd_data.shape[0]),
				y=fevd_data[:, j],
				mode='lines',
				stackgroup='one',
				name=f'مساهمة {source_name}'
			))
		fig.update_layout(
			title=f'تجزئة التباين: {name}',
			xaxis_title="الفترات",
			yaxis_title="نسبة التباين",
			height=500,
			template="plotly_white"
		)
		figs.append(fig)
	return figs


# رسم العلاقات بين المتغيرات
def plot_relationships(data):
	fig = px.scatter_matrix(
		data,
		dimensions=data.columns,
		height=800,
		template="plotly_white"
	)
	fig.update_layout(title="العلاقات بين المتغيرات")
	return fig


# رسم البواقي
def plot_residuals(residuals):
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=np.arange(len(residuals)), y=residuals, mode='lines', name='البواقي'))
	fig.update_layout(
		title="البواقي من انحدار التكامل المشترك",
		xaxis_title="الزمن",
		yaxis_title="القيمة",
		height=400,
		template="plotly_white"
	)
	return fig


# رسم توضيحي لنموذج VAR
def plot_var_illustration():
	fig = go.Figure()

	# متغير أول
	x = np.arange(50)
	y1 = np.cumsum(np.random.normal(0, 1, 50))

	# متغير ثاني متأثر بالأول
	y2 = np.zeros(50)
	for i in range(2, 50):
		y2[i] = 0.7 * y2[i - 1] + 0.3 * y1[i - 2] + np.random.normal(0, 0.5)

	fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='y1'))
	fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='y2'))

	# إضافة أسهم لتوضيح التأثير المتبادل
	fig.add_annotation(
		x=25, y=y1[25],
		ax=25, ay=y2[27],
		axref="x", ayref="y",
		text="",
		showarrow=True,
		arrowhead=2,
		arrowsize=1,
		arrowwidth=2,
		arrowcolor="#636EFA"
	)

	fig.add_annotation(
		x=35, y=y2[35],
		ax=35, ay=y1[37],
		axref="x", ayref="y",
		text="",
		showarrow=True,
		arrowhead=2,
		arrowsize=1,
		arrowwidth=2,
		arrowcolor="#EF553B"
	)

	fig.update_layout(
		title="توضيح نموذج VAR: كل متغير يؤثر على الآخر",
		xaxis_title="الزمن",
		yaxis_title="القيمة",
		height=400,
		template="plotly_white"
	)

	return fig


# رسم توضيحي لنموذج VECM
def plot_vecm_illustration():
	fig = go.Figure()

	# متغيرين متكاملين مشتركًا
	np.random.seed(42)
	x = np.arange(100)

	# المتغير المشترك
	common = np.cumsum(np.random.normal(0, 1, 100))

	# المتغيرات
	y1 = common + np.random.normal(0, 1, 100)
	y2 = 2 * common + 5 + np.random.normal(0, 1, 100)

	# عرض المتغيرات
	fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='y1'))
	fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='y2'))

	# عرض مسار التوازن
	eq_line = 2 * y1 + 5
	fig.add_trace(go.Scatter(x=x, y=eq_line, mode='lines', line=dict(dash='dash'), name='مسار التوازن'))

	# إضافة أسهم توضيحية
	for i in [30, 50, 70]:
		# أسهم تشير إلى العودة للتوازن
		if y2[i] < eq_line[i]:  # إذا كان تحت خط التوازن
			fig.add_annotation(
				x=i, y=y2[i],
				ax=i, ay=y2[i] + (eq_line[i] - y2[i]) * 0.3,
				axref="x", ayref="y",
				text="",
				showarrow=True,
				arrowhead=1,
				arrowsize=1.5,
				arrowwidth=2,
				arrowcolor="#00CC96"
			)
		else:  # إذا كان فوق خط التوازن
			fig.add_annotation(
				x=i, y=y2[i],
				ax=i, ay=y2[i] - (y2[i] - eq_line[i]) * 0.3,
				axref="x", ayref="y",
				text="",
				showarrow=True,
				arrowhead=1,
				arrowsize=1.5,
				arrowwidth=2,
				arrowcolor="#00CC96"
			)

	fig.update_layout(
		title="توضيح نموذج VECM: التصحيح نحو التوازن طويل الأجل",
		xaxis_title="الزمن",
		yaxis_title="القيمة",
		height=500,
		template="plotly_white"
	)

	return fig


# بداية التطبيق
st.markdown('<h1 class="main-header rtl">تحليل السلاسل الزمنية متعددة المتغيرات: VAR, VECM والتكامل المشترك</h1>',
			unsafe_allow_html=True)

# القائمة الجانبية
menu = st.sidebar.radio(
	"المحتويات",
	["مقدمة",
	 "استقرارية السلاسل الزمنية",
	 "اختبار انجل-جرانجر للتكامل المشترك",
	 "اختبار جوهانسون للتكامل المشترك",
	 "نموذج VAR",
	 "نموذج VECM",
	 "السببية لجرانجر",
	 "تطبيق عملي",
	 "الملخص والاستنتاجات"]
)

# إنشاء بيانات نموذجية
data = generate_sample_data()

# 1. مقدمة
if menu == "مقدمة":
	st.markdown('<div class="rtl">', unsafe_allow_html=True)
	st.markdown('<h2 class="sub-header">مقدمة حول السلاسل الزمنية متعددة المتغيرات</h2>', unsafe_allow_html=True)

	st.markdown('<div class="concept">', unsafe_allow_html=True)
	st.write("""
    تُعدّ نماذج تحليل السلاسل الزمنية متعددة المتغيرات من الأدوات الإحصائية الأساسية في الاقتصاد القياسي، وتُستخدم لدراسة العلاقات الديناميكية بين المتغيرات الاقتصادية. من أهم هذه النماذج:

    1. **نموذج متجه الانحدار الذاتي (Vector Autoregression - VAR)**: يُستخدم لدراسة العلاقات الديناميكية بين المتغيرات المستقرة.

    2. **نموذج تصحيح الخطأ المتجهي (Vector Error Correction Model - VECM)**: يُستخدم عندما توجد علاقة تكامل مشترك بين المتغيرات.

    قبل تطبيق هذه النماذج، يجب إجراء عدة اختبارات للتحقق من خصائص البيانات واختيار النموذج المناسب.
    """)
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<h3>مفهوم التكامل المشترك (Cointegration)</h3>', unsafe_allow_html=True)
	st.markdown('<div class="formula-box">', unsafe_allow_html=True)
	st.write("""
    التكامل المشترك هو خاصية إحصائية للسلاسل الزمنية تشير إلى وجود علاقة توازنية طويلة الأجل بين متغيرين أو أكثر. بشكل رسمي، إذا كانت هناك مجموعة من المتغيرات المتكاملة من الدرجة الأولى I(1)، وكان هناك تركيبة خطية منها مستقرة I(0)، فإنها تكون متكاملة تكاملاً مشتركاً.
    """)

	st.latex(r'''
    \text{إذا كان } X_t \sim I(1), Y_t \sim I(1) \text{ و } \exists \beta : (Y_t - \beta X_t) \sim I(0)
    ''')

	st.latex(r'''
    \text{فإن } X_t \text{ و } Y_t \text{ متكاملان تكاملاً مشتركاً.}
    ''')
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<div class="application">', unsafe_allow_html=True)
	st.write("""
    **أهمية التكامل المشترك في الاقتصاد:**

    يُعدّ مفهوم التكامل المشترك ذا أهمية كبيرة في الاقتصاد لأنه يسمح بتحليل العلاقات طويلة الأجل بين المتغيرات الاقتصادية. على سبيل المثال:

    - العلاقة بين الاستهلاك والدخل
    - العلاقة بين أسعار السلع في أسواق مختلفة
    - العلاقة بين أسعار الفائدة قصيرة وطويلة الأجل
    - العلاقة بين أسعار الأسهم وأرباح الشركات
    """)
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<div class="note">', unsafe_allow_html=True)
	st.write("""
    **ملاحظة مهمة:**

    تحديد وجود التكامل المشترك بين المتغيرات هو الخطوة الحاسمة في اختيار النموذج المناسب:

    - إذا كانت المتغيرات مستقرة I(0): نستخدم نموذج VAR
    - إذا كانت المتغيرات متكاملة من الدرجة الأولى I(1) وغير متكاملة تكاملاً مشتركاً: نستخدم نموذج VAR مع الفروق الأولى
    - إذا كانت المتغيرات متكاملة من الدرجة الأولى I(1) ومتكاملة تكاملاً مشتركاً: نستخدم نموذج VECM
    """)
	st.markdown('</div>', unsafe_allow_html=True)

	# عرض البيانات النموذجية
	st.markdown('<h3>البيانات النموذجية المستخدمة في التطبيق</h3>', unsafe_allow_html=True)
	st.write("""
    سنستخدم بيانات اصطناعية للتوضيح:
    - `y1`: سلسلة زمنية متكاملة من الدرجة الأولى I(1)
    - `y2`: سلسلة زمنية متكاملة من الدرجة الأولى I(1) ومتكاملة تكاملاً مشتركاً مع `y1`
    - `y3`: سلسلة زمنية متكاملة من الدرجة الأولى I(1) ومستقلة عن المتغيرات الأخرى
    """)

	st.dataframe(data.head())

	# رسم بياني للسلاسل الزمنية
	st.plotly_chart(plot_time_series(data))

	st.markdown('<div class="conclusion">', unsafe_allow_html=True)
	st.write("""
    خلال هذا التطبيق، سنقوم بتحليل هذه البيانات خطوة بخطوة:
    1. اختبار استقرارية السلاسل الزمنية
    2. اختبار التكامل المشترك باستخدام طريقة انجل-جرانجر وطريقة جوهانسون
    3. تقدير نماذج VAR و VECM
    4. تحليل دوال الاستجابة النبضية وتجزئة التباين
    5. اختبار السببية لجرانجر
    """)
	st.markdown('</div>', unsafe_allow_html=True)
	st.markdown('</div>', unsafe_allow_html=True)

# 2. استقرارية السلاسل الزمنية
elif menu == "استقرارية السلاسل الزمنية":
	st.markdown('<div class="rtl">', unsafe_allow_html=True)
	st.markdown('<h2 class="sub-header">استقرارية السلاسل الزمنية</h2>', unsafe_allow_html=True)

	st.markdown('<div class="concept">', unsafe_allow_html=True)
	st.write("""
    تُعدّ استقرارية السلاسل الزمنية (Stationarity) من المفاهيم الأساسية في تحليل السلاسل الزمنية. السلسلة الزمنية تكون مستقرة إذا كانت خصائصها الإحصائية (المتوسط، التباين، التغاير) ثابتة عبر الزمن.
    """)

	st.markdown('<h3>التعريف الرسمي للاستقرارية</h3>', unsafe_allow_html=True)
	st.latex(r'''
    \text{السلسلة } Y_t \text{ تكون مستقرة إذا:}
    ''')

	st.latex(r'''
    \begin{align}
    &1. \ E[Y_t] = \mu \text{ (المتوسط ثابت)} \\
    &2. \ Var[Y_t] = \sigma^2 \text{ (التباين ثابت)} \\
    &3. \ Cov[Y_t, Y_{t-h}] = \gamma_h \text{ (التغاير يعتمد فقط على فترة الإبطاء } h \text{)}
    \end{align}
    ''')
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<h3>أنواع الاستقرارية وعدم الاستقرارية</h3>', unsafe_allow_html=True)
	st.write("""
    هناك عدة مستويات للاستقرارية وعدم الاستقرارية:

    1. **السلسلة المستقرة (I(0))**: تتذبذب حول متوسط ثابت وتباينها محدود.

    2. **السلسلة المتكاملة من الدرجة الأولى (I(1))**: تحتوي على جذر وحدة، وتصبح مستقرة بعد أخذ الفروق الأولى.

    3. **السلسلة المتكاملة من الدرجة d (I(d))**: تصبح مستقرة بعد أخذ الفروق d مرة.
    """)

	st.markdown('<div class="formula-box">', unsafe_allow_html=True)
	st.write("**الفروق الأولى للسلسلة الزمنية:**")
	st.latex(r'\Delta Y_t = Y_t - Y_{t-1}')

	st.write("**السلسلة المتكاملة من الدرجة الأولى I(1):**")
	st.latex(r'Y_t \sim I(1) \Rightarrow \Delta Y_t \sim I(0)')
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<h3>اختبار جذر الوحدة (Augmented Dickey-Fuller Test)</h3>', unsafe_allow_html=True)
	st.markdown('<div class="formula-box">', unsafe_allow_html=True)
	st.write("""
    اختبار ديكي-فولر الموسع (ADF) هو الاختبار الأكثر شيوعاً لفحص استقرارية السلسلة الزمنية. يختبر وجود جذر وحدة في السلسلة.

    النموذج المختبر:
    """)

	st.latex(r'\Delta Y_t = \alpha + \beta t + \gamma Y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta Y_{t-i} + \varepsilon_t')

	st.write("""
    الفرضية الصفرية والبديلة:
    """)

	st.latex(r'''
    \begin{align}
    &H_0: \gamma = 0 \text{ (يوجد جذر وحدة، السلسلة غير مستقرة)} \\
    &H_1: \gamma < 0 \text{ (لا يوجد جذر وحدة، السلسلة مستقرة)}
    \end{align}
    ''')
	st.markdown('</div>', unsafe_allow_html=True)

	# تطبيق اختبار ADF على البيانات النموذجية
	st.markdown('<h3>تطبيق اختبار جذر الوحدة على البيانات</h3>', unsafe_allow_html=True)

	for col in data.columns:
		st.subheader(f"نتائج اختبار ADF للسلسلة {col}")
		result = run_adf_test(data[col])
		st.write(f"إحصائية الاختبار: {result['Test Statistic']:.4f}")
		st.write(f"قيمة p: {result['p-value']:.4f}")
		st.write(f"القيم الحرجة:")
		for key, value in result['Critical Values'].items():
			st.write(f"   {key}: {value:.4f}")

		if result['Stationary']:
			st.success(f"السلسلة {col} مستقرة (نرفض الفرضية الصفرية)")
		else:
			st.error(f"السلسلة {col} غير مستقرة (لا نستطيع رفض الفرضية الصفرية)")

	# اختبار ADF على الفروق الأولى
	st.markdown('<h3>اختبار جذر الوحدة على الفروق الأولى</h3>', unsafe_allow_html=True)

	diff_data = data.diff().dropna()

	for col in diff_data.columns:
		st.subheader(f"نتائج اختبار ADF للفروق الأولى للسلسلة {col}")
		result = run_adf_test(diff_data[col])
		st.write(f"إحصائية الاختبار: {result['Test Statistic']:.4f}")
		st.write(f"قيمة p: {result['p-value']:.4f}")
		st.write(f"القيم الحرجة:")
		for key, value in result['Critical Values'].items():
			st.write(f"   {key}: {value:.4f}")

		if result['Stationary']:
			st.success(f"الفروق الأولى للسلسلة {col} مستقرة (نرفض الفرضية الصفرية)")
		else:
			st.error(f"الفروق الأولى للسلسلة {col} غير مستقرة (لا نستطيع رفض الفرضية الصفرية)")

	st.markdown('<div class="application">', unsafe_allow_html=True)
	st.write("""
    **الاستنتاج من اختبارات الاستقرارية:**

    1. جميع السلاسل الزمنية الأصلية (y1, y2, y3) غير مستقرة (متكاملة من الدرجة الأولى I(1)).

    2. الفروق الأولى لجميع السلاسل مستقرة (I(0)).

    هذا يعني أن السلاسل متكاملة من الدرجة الأولى I(1)، مما يجعلها مرشحة للتكامل المشترك. الخطوة التالية هي اختبار وجود علاقات تكامل مشترك بين هذه المتغيرات باستخدام اختبارات انجل-جرانجر وجوهانسون.
    """)
	st.markdown('</div>', unsafe_allow_html=True)

	# رسم السلاسل الزمنية والفروق الأولى
	st.subheader("مقارنة بين السلاسل الزمنية الأصلية والفروق الأولى")

	col1, col2 = st.columns(2)

	with col1:
		st.plotly_chart(plot_time_series(data), use_container_width=True)
		st.write("السلاسل الزمنية الأصلية (غير مستقرة)")

	with col2:
		st.plotly_chart(plot_time_series(diff_data), use_container_width=True)
		st.write("الفروق الأولى للسلاسل الزمنية (مستقرة)")

	st.markdown('<div class="note">', unsafe_allow_html=True)
	st.write("""
    **ملاحظة مهمة:**

    عندما نتعامل مع سلاسل زمنية متكاملة من الدرجة الأولى I(1)، فإن استخدام نموذج VAR على المستويات الأصلية قد يؤدي إلى استنتاجات خاطئة (مثل الانحدار الزائف). الخيارات المتاحة:

    1. استخدام نموذج VAR مع الفروق الأولى للمتغيرات (إذا لم يكن هناك تكامل مشترك).

    2. استخدام نموذج VECM (إذا كان هناك تكامل مشترك).

    لذلك، اختبار التكامل المشترك هو الخطوة التالية المهمة في تحليلنا.
    """)
	st.markdown('</div>', unsafe_allow_html=True)
	st.markdown('</div>', unsafe_allow_html=True)

# 3. اختبار انجل-جرانجر للتكامل المشترك
elif menu == "اختبار انجل-جرانجر للتكامل المشترك":
	st.markdown('<div class="rtl">', unsafe_allow_html=True)
	st.markdown('<h2 class="sub-header">اختبار انجل-جرانجر للتكامل المشترك</h2>', unsafe_allow_html=True)

	st.markdown('<div class="concept">', unsafe_allow_html=True)
	st.write("""
    اختبار انجل-جرانجر هو أحد الطرق الأساسية لاختبار التكامل المشترك بين متغيرين I(1). يعتمد على مبدأ أن المتغيرات المتكاملة تكاملاً مشتركاً ستكون لها علاقة توازنية طويلة الأجل، وأن الانحرافات عن هذا التوازن (البواقي) ستكون مستقرة I(0).
    """)
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<h3>منهجية اختبار انجل-جرانجر</h3>', unsafe_allow_html=True)
	st.markdown('<div class="formula-box">', unsafe_allow_html=True)
	st.write("تتكون طريقة انجل-جرانجر من خطوتين أساسيتين:")

	st.write("**الخطوة 1**: تقدير معادلة التكامل المشترك (الانحدار طويل الأمد)")
	st.latex(r'Y_t = \alpha + \beta X_t + u_t')

	st.write("**الخطوة 2**: اختبار استقرارية البواقي باستخدام اختبار ADF")
	st.latex(r'\hat{u}_t = Y_t - \hat{\alpha} - \hat{\beta} X_t')

	st.write("إذا كانت البواقي $\\hat{u}_t$ مستقرة I(0)، فإن المتغيرات متكاملة تكاملاً مشتركاً.")
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<h3>الفرضيات في اختبار انجل-جرانجر</h3>', unsafe_allow_html=True)
	st.markdown('<div class="formula-box">', unsafe_allow_html=True)
	st.latex(r'''
    \begin{align}
    &H_0: \text{لا يوجد تكامل مشترك (البواقي غير مستقرة)} \\
    &H_1: \text{يوجد تكامل مشترك (البواقي مستقرة)}
    \end{align}
    ''')
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<div class="caution">', unsafe_allow_html=True)
	st.write("""
    **محددات وقيود اختبار انجل-جرانجر:**

    1. **اتجاه العلاقة**: يفترض الاختبار اتجاه معين للعلاقة (Y على X). عند عكس الاتجاه، قد نحصل على نتائج مختلفة.

    2. **ثنائية المتغيرات**: الاختبار مصمم لمتغيرين فقط، ولا يمكن استخدامه بشكل مباشر مع أكثر من متغيرين.

    3. **وجود متجه تكامل مشترك واحد فقط**: لا يمكن للاختبار اكتشاف وجود متجهات تكامل مشترك متعددة.

    4. **القيم الحرجة**: القيم الحرجة لاختبار ADF على البواقي تختلف عن القيم القياسية لاختبار ADF العادي.
    """)
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<h3>تطبيق اختبار انجل-جرانجر على البيانات</h3>', unsafe_allow_html=True)

	# اختبار انجل-جرانجر للتكامل المشترك
	pairs = [('y1', 'y2'), ('y1', 'y3'), ('y2', 'y3')]

	for pair in pairs:
		st.subheader(f"اختبار التكامل المشترك بين {pair[0]} و {pair[1]}")

		# اختبار الاتجاه الأول
		st.write(f"**الاتجاه الأول: {pair[0]} → {pair[1]}**")
		eg_result = run_engle_granger(data[pair[0]], data[pair[1]])

		st.write(f"معامل الانحدار: {eg_result['Coefficient']:.4f}")
		st.write(f"إحصائية اختبار ADF للبواقي: {eg_result['ADF Test Statistic']:.4f}")
		st.write(f"قيمة p للبواقي: {eg_result['ADF p-value']:.4f}")

		if eg_result['Cointegration']:
			st.success(f"يوجد تكامل مشترك بين {pair[0]} و {pair[1]} (نرفض الفرضية الصفرية)")
		else:
			st.error(f"لا يوجد تكامل مشترك بين {pair[0]} و {pair[1]} (لا نستطيع رفض الفرضية الصفرية)")

		# رسم البواقي
		st.plotly_chart(plot_residuals(eg_result['Residuals']), use_container_width=True)

		# اختبار الاتجاه الثاني
		st.write(f"**الاتجاه الثاني: {pair[1]} → {pair[0]}**")
		eg_result_rev = run_engle_granger(data[pair[1]], data[pair[0]])

		st.write(f"معامل الانحدار: {eg_result_rev['Coefficient']:.4f}")
		st.write(f"إحصائية اختبار ADF للبواقي: {eg_result_rev['ADF Test Statistic']:.4f}")
		st.write(f"قيمة p للبواقي: {eg_result_rev['ADF p-value']:.4f}")

		if eg_result_rev['Cointegration']:
			st.success(f"يوجد تكامل مشترك بين {pair[1]} و {pair[0]} (نرفض الفرضية الصفرية)")
		else:
			st.error(f"لا يوجد تكامل مشترك بين {pair[1]} و {pair[0]} (لا نستطيع رفض الفرضية الصفرية)")

		# رسم البواقي
		st.plotly_chart(plot_residuals(eg_result_rev['Residuals']), use_container_width=True)

	st.markdown('<div class="application">', unsafe_allow_html=True)
	st.write("""
    **الاستنتاج من اختبار انجل-جرانجر:**

    1. **العلاقة بين y1 و y2**: يوجد تكامل مشترك بين هذين المتغيرين في كلا الاتجاهين، مما يؤكد وجود علاقة توازنية طويلة الأجل بينهما.

    2. **العلاقة بين y1 و y3**: لا يوجد تكامل مشترك بين هذين المتغيرين، مما يشير إلى أنهما لا يتشاركان مساراً مشتركاً على المدى الطويل.

    3. **العلاقة بين y2 و y3**: لا يوجد تكامل مشترك بين هذين المتغيرين، مما يؤكد أن y3 مستقلة عن المتغيرات الأخرى.

    هذه النتائج تتسق مع طريقة توليد البيانات، حيث تم إنشاء y1 و y2 بحيث يكون بينهما علاقة تكامل مشترك، بينما y3 هي سلسلة مستقلة.
    """)
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<h3>متى يستخدم اختبار انجل-جرانجر؟</h3>', unsafe_allow_html=True)
	st.markdown('<div class="note">', unsafe_allow_html=True)
	st.write("""
    **الحالات المناسبة لاستخدام اختبار انجل-جرانجر:**

    1. **متغيران فقط**: عندما يكون لدينا متغيران فقط للاختبار.

    2. **السيناريوهات البسيطة**: عندما نتوقع وجود متجه تكامل مشترك واحد فقط.

    3. **الاختبارات الأولية**: كخطوة أولية قبل اللجوء إلى اختبارات أكثر تعقيداً.

    **الحالات غير المناسبة:**

    1. **نظام متعدد المتغيرات**: عندما يكون لدينا أكثر من متغيرين.

    2. **وجود متجهات تكامل مشترك متعددة**: عندما يكون من الممكن وجود أكثر من متجه تكامل مشترك.

    3. **عندما يكون هناك غموض في اتجاه العلاقة**: في هذه الحالة، يفضل استخدام اختبار جوهانسون.
    """)
	st.markdown('</div>', unsafe_allow_html=True)
	st.markdown('</div>', unsafe_allow_html=True)

# 4. اختبار جوهانسون للتكامل المشترك
elif menu == "اختبار جوهانسون للتكامل المشترك":
	st.markdown('<div class="rtl">', unsafe_allow_html=True)
	st.markdown('<h2 class="sub-header">اختبار جوهانسون للتكامل المشترك</h2>', unsafe_allow_html=True)

	st.markdown('<div class="concept">', unsafe_allow_html=True)
	st.write("""
    اختبار جوهانسون هو طريقة أكثر تقدماً لاختبار التكامل المشترك، ويعتمد على نموذج متجه الانحدار الذاتي (VAR). يتميز عن اختبار انجل-جرانجر بأنه:

    1. يمكنه التعامل مع أكثر من متغيرين في وقت واحد.
    2. يمكنه اكتشاف وجود متجهات تكامل مشترك متعددة.
    3. لا يتطلب تحديد اتجاه العلاقة مسبقاً.
    """)
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<h3>منهجية اختبار جوهانسون</h3>', unsafe_allow_html=True)
	st.markdown('<div class="formula-box">', unsafe_allow_html=True)
	st.write("""
    يعتمد اختبار جوهانسون على تقدير نموذج VECM والبحث عن رتبة مصفوفة المعاملات Π:
    """)

	st.latex(r'\Delta Y_t = \Pi Y_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta Y_{t-i} + \mu + \varepsilon_t')

	st.write("""
    حيث:
    - Π هي مصفوفة المعاملات التي تحتوي على معلومات التكامل المشترك.
    - رتبة Π تحدد عدد متجهات التكامل المشترك.

    يمكن تفكيك المصفوفة Π كالتالي:
    """)

	st.latex(r'\Pi = \alpha \beta^\prime')

	st.write("""
    حيث:
    - β هي مصفوفة متجهات التكامل المشترك.
    - α هي مصفوفة معاملات التعديل (معاملات تصحيح الخطأ).
    """)
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<h3>اختبارات جوهانسون</h3>', unsafe_allow_html=True)
	st.markdown('<div class="formula-box">', unsafe_allow_html=True)
	st.write("""
    يقدم اختبار جوهانسون نوعين من الاختبارات:
    """)

	st.write("**1. اختبار الأثر (Trace Test):**")
	st.latex(r'\lambda_{trace}(r) = -T \sum_{i=r+1}^{n} \ln(1 - \hat{\lambda}_i)')

	st.write("""
    الفرضية الصفرية: عدد متجهات التكامل المشترك $\\leq r$
    الفرضية البديلة: عدد متجهات التكامل المشترك $> r$
    """)

	st.write("**2. اختبار القيمة الذاتية العظمى (Maximum Eigenvalue Test):**")
	st.latex(r'\lambda_{max}(r, r+1) = -T \ln(1 - \hat{\lambda}_{r+1})')

	st.write("""
    الفرضية الصفرية: عدد متجهات التكامل المشترك = $r$
    الفرضية البديلة: عدد متجهات التكامل المشترك = $r+1$
    """)
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<h3>تفسير نتائج اختبار جوهانسون</h3>', unsafe_allow_html=True)
	st.write("""
    تفسير النتائج يعتمد على مقارنة إحصاءات الاختبار مع القيم الحرجة:

    - إذا كانت إحصاءة الاختبار > القيمة الحرجة: نرفض الفرضية الصفرية.
    - إذا كانت إحصاءة الاختبار < القيمة الحرجة: لا نستطيع رفض الفرضية الصفرية.

    يتم الاختبار بشكل متسلسل بدءاً من r = 0، r = 1، ... الخ. نتوقف عند أول فرضية صفرية لا نستطيع رفضها.
    """)

	st.markdown('<div class="note">', unsafe_allow_html=True)
	st.write("""
    **تفسير رتبة التكامل المشترك:**

    - **r = 0**: لا يوجد تكامل مشترك بين المتغيرات.
    - **r = 1**: يوجد متجه تكامل مشترك واحد.
    - **r = 2**: يوجد متجهان للتكامل المشترك.
    - وهكذا...

    الحد الأقصى لعدد متجهات التكامل المشترك هو n-1 حيث n هو عدد المتغيرات.
    """)
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<h3>تطبيق اختبار جوهانسون على البيانات</h3>', unsafe_allow_html=True)

	# اختبار جوهانسون للمجموعات المختلفة من المتغيرات
	variable_groups = [
		['y1', 'y2'],  # نتوقع وجود تكامل مشترك
		['y1', 'y3'],  # لا نتوقع وجود تكامل مشترك
		['y2', 'y3'],  # لا نتوقع وجود تكامل مشترك
		['y1', 'y2', 'y3']  # نتوقع وجود متجه تكامل مشترك واحد
	]

	for group in variable_groups:
		st.subheader(f"اختبار جوهانسون للمتغيرات: {', '.join(group)}")

		# تطبيق اختبار جوهانسون
		subset_data = data[group]
		johansen_result = run_johansen(subset_data)

		if isinstance(johansen_result, str):
			st.warning(johansen_result)
			continue

		# عرض نتائج اختبار الأثر
		st.write("**نتائج اختبار الأثر (Trace Test):**")
		results_data = {
			'رتبة التكامل المشترك': [f'r ≤ {i}' for i in range(len(group))],
			'إحصاءة الأثر': johansen_result['Trace Statistics'],
			'القيمة الحرجة (95%)': johansen_result['Trace Critical Values (95%)']
		}
		results_df = pd.DataFrame(results_data)
		st.dataframe(results_df)

		# عرض نتائج اختبار القيمة الذاتية العظمى
		st.write("**نتائج اختبار القيمة الذاتية العظمى (Max Eigenvalue Test):**")
		results_data = {
			'رتبة التكامل المشترك': [f'r = {i}' for i in range(len(group))],
			'إحصاءة القيمة الذاتية العظمى': johansen_result['Max Eigenvalue Statistics'],
			'القيمة الحرجة (95%)': johansen_result['Max Eigenvalue Critical Values (95%)']
		}
		results_df = pd.DataFrame(results_data)
		st.dataframe(results_df)

		# تحديد رتبة التكامل المشترك
		st.write(f"**رتبة التكامل المشترك (اختبار الأثر): {johansen_result['Cointegration Rank (Trace)']}**")
		st.write(
			f"**رتبة التكامل المشترك (اختبار القيمة الذاتية العظمى): {johansen_result['Cointegration Rank (Max Eigenvalue)']}**")

		if johansen_result['Cointegration Rank (Trace)'] > 0:
			st.success(f"يوجد تكامل مشترك بين المتغيرات {', '.join(group)} وفقاً لاختبار الأثر")
		else:
			st.error(f"لا يوجد تكامل مشترك بين المتغيرات {', '.join(group)} وفقاً لاختبار الأثر")

		if johansen_result['Cointegration Rank (Max Eigenvalue)'] > 0:
			st.success(f"يوجد تكامل مشترك بين المتغيرات {', '.join(group)} وفقاً لاختبار القيمة الذاتية العظمى")
		else:
			st.error(f"لا يوجد تكامل مشترك بين المتغيرات {', '.join(group)} وفقاً لاختبار القيمة الذاتية العظمى")

	st.markdown('<div class="application">', unsafe_allow_html=True)
	st.write("""
    **الاستنتاج من اختبار جوهانسون:**

    1. **المتغيرات y1 و y2**: يوجد متجه تكامل مشترك واحد، مما يؤكد نتائج اختبار انجل-جرانجر.

    2. **المتغيرات y1 و y3**: لا يوجد تكامل مشترك، مما يتسق مع نتائج اختبار انجل-جرانجر.

    3. **المتغيرات y2 و y3**: لا يوجد تكامل مشترك، مما يتسق مع نتائج اختبار انجل-جرانجر.

    4. **المتغيرات y1 و y2 و y3**: يوجد متجه تكامل مشترك واحد، وهذا يتفق مع الطريقة التي تم بها توليد البيانات، حيث يوجد علاقة تكامل مشترك بين y1 و y2 فقط.
    """)
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<h3>مقارنة بين اختبار انجل-جرانجر واختبار جوهانسون</h3>', unsafe_allow_html=True)
	col1, col2 = st.columns(2)

	with col1:
		st.markdown('<div class="application">', unsafe_allow_html=True)
		st.write("**مزايا اختبار انجل-جرانجر:**")
		st.write("""
        - بسيط وسهل التطبيق
        - مناسب للتحليل الثنائي المتغيرات
        - سهل التفسير
        """)

		st.write("**عيوب اختبار انجل-جرانجر:**")
		st.write("""
        - لا يناسب النظم متعددة المتغيرات
        - لا يكتشف متجهات تكامل مشترك متعددة
        - حساس لاتجاه الانحدار
        - لا يأخذ في الاعتبار الديناميكية قصيرة الأجل
        """)
		st.markdown('</div>', unsafe_allow_html=True)

	with col2:
		st.markdown('<div class="application">', unsafe_allow_html=True)
		st.write("**مزايا اختبار جوهانسون:**")
		st.write("""
        - يتعامل مع أنظمة متعددة المتغيرات
        - يكتشف متجهات تكامل مشترك متعددة
        - لا يتطلب تحديد اتجاه العلاقة
        - يأخذ في الاعتبار الديناميكية قصيرة الأجل
        """)

		st.write("**عيوب اختبار جوهانسون:**")
		st.write("""
        - أكثر تعقيداً من الناحية الحسابية
        - يتطلب عينات كبيرة للحصول على نتائج موثوقة
        - حساس لتحديد عدد فترات الإبطاء
        - افتراضات أكثر حول توزيع الأخطاء
        """)
		st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<div class="note">', unsafe_allow_html=True)
	st.write("""
    **متى تستخدم كل اختبار؟**

    - **اختبار انجل-جرانجر**: عندما تتعامل مع متغيرين فقط وتتوقع وجود متجه تكامل مشترك واحد فقط.

    - **اختبار جوهانسون**: في معظم الحالات الأخرى، خاصة عندما:
      - تتعامل مع أكثر من متغيرين.
      - تتوقع وجود أكثر من متجه تكامل مشترك.
      - تكون هناك علاقات ديناميكية معقدة بين المتغيرات.
      - تكون لديك عينة كبيرة بما فيه الكفاية.
    """)
	st.markdown('</div>', unsafe_allow_html=True)
	st.markdown('</div>', unsafe_allow_html=True)

# 5. نموذج VAR
elif menu == "نموذج VAR":
	st.markdown('<div class="rtl">', unsafe_allow_html=True)
	st.markdown('<h2 class="sub-header">نموذج متجه الانحدار الذاتي (Vector Autoregression - VAR)</h2>',
				unsafe_allow_html=True)

	st.markdown('<div class="concept">', unsafe_allow_html=True)
	st.write("""
    نموذج متجه الانحدار الذاتي (VAR) هو امتداد لنماذج الانحدار الذاتي أحادية المتغير إلى بيئة متعددة المتغيرات. يعامل النموذج كل متغير كدالة لقيمه السابقة وقيم المتغيرات الأخرى السابقة في النظام.
    """)
	st.markdown('</div>', unsafe_allow_html=True)

	# توضيح رسومي لنموذج VAR
	st.plotly_chart(plot_var_illustration(), use_container_width=True)

	st.markdown('<h3>الصيغة الرياضية لنموذج VAR</h3>', unsafe_allow_html=True)
	st.markdown('<div class="formula-box">', unsafe_allow_html=True)
	st.write("نموذج VAR من الدرجة p يمكن كتابته كالتالي:")

	st.latex(r'Y_t = c + A_1 Y_{t-1} + A_2 Y_{t-2} + ... + A_p Y_{t-p} + \varepsilon_t')

	st.write("حيث:")
	st.latex(r'''
    \begin{align}
    &Y_t \text{ متجه المتغيرات في الزمن } t \\
    &c \text{ متجه الثوابت} \\
    &A_i \text{ مصفوفات المعاملات} \\
    &\varepsilon_t \text{ متجه الأخطاء العشوائية}
    \end{align}
    ''')

	st.write("على سبيل المثال، نموذج VAR(1) لمتغيرين:")

	st.latex(r'''
    \begin{bmatrix} y_{1t} \\ y_{2t} \end{bmatrix} = 
    \begin{bmatrix} c_1 \\ c_2 \end{bmatrix} + 
    \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}
    \begin{bmatrix} y_{1,t-1} \\ y_{2,t-1} \end{bmatrix} + 
    \begin{bmatrix} \varepsilon_{1t} \\ \varepsilon_{2t} \end{bmatrix}
    ''')
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<h3>شروط تطبيق نموذج VAR</h3>', unsafe_allow_html=True)
	st.markdown('<div class="note">', unsafe_allow_html=True)
	st.write("""
    **الشروط الأساسية لتطبيق نموذج VAR:**

    1. **استقرارية المتغيرات**: يجب أن تكون جميع المتغيرات مستقرة I(0).

    2. **عدم وجود تكامل مشترك**: إذا كانت المتغيرات متكاملة من الدرجة الأولى I(1) ومتكاملة تكاملاً مشتركاً، فإن نموذج VECM هو الأنسب.

    3. **استقلالية الأخطاء**: يجب أن تكون الأخطاء في النموذج مستقلة عبر الزمن (عدم وجود ارتباط ذاتي).

    4. **ثبات التباين**: يجب أن يكون تباين الأخطاء ثابتاً عبر الزمن.
    """)
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<h3>خطوات تقدير نموذج VAR</h3>', unsafe_allow_html=True)
	st.write("""
    1. **اختبار استقرارية المتغيرات**: باستخدام اختبارات جذر الوحدة مثل ADF.

    2. **تحديد عدد فترات الإبطاء المثلى (p)**: باستخدام معايير المعلومات مثل AIC و BIC و HQ.

    3. **تقدير النموذج**: باستخدام طرق التقدير المناسبة مثل OLS.

    4. **اختبار صحة النموذج**: التحقق من استقلالية الأخطاء وثبات التباين.

    5. **تحليل النتائج**: تفسير المعاملات، دوال الاستجابة النبضية، تجزئة التباين، إلخ.
    """)

	st.markdown('<h3>أدوات تحليل نموذج VAR</h3>', unsafe_allow_html=True)

	st.markdown('<div class="concept">', unsafe_allow_html=True)
	st.write("**1. دالة الاستجابة النبضية (Impulse Response Function):**")
	st.write("""
    تقيس استجابة المتغيرات في النظام لصدمة قدرها وحدة انحراف معياري واحدة في أحد المتغيرات، مع ثبات المتغيرات الأخرى.
    """)

	st.latex(r'''
    \frac{\partial y_{i,t+s}}{\partial \varepsilon_{j,t}} = \text{استجابة المتغير } i \text{ بعد } s \text{ فترات لصدمة في المتغير } j
    ''')
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<div class="concept">', unsafe_allow_html=True)
	st.write("**2. تجزئة التباين (Variance Decomposition):**")
	st.write("""
    تقيس النسبة المئوية لتباين الخطأ في التنبؤ للمتغير الذي يمكن أن يعزى إلى صدمات في كل متغير من متغيرات النظام.
    """)

	st.latex(r'''
    \theta_{ij}(h) = \frac{\sum_{s=0}^{h-1} (\text{مربع تأثير صدمة } j \text{ على } i \text{ بعد } s \text{ فترات})}{\text{تباين الخطأ الكلي في التنبؤ للمتغير } i \text{ بعد } h \text{ فترات}}
    ''')
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<div class="concept">', unsafe_allow_html=True)
	st.write("**3. اختبار السببية لجرانجر (Granger Causality):**")
	st.write("""
    يختبر ما إذا كانت القيم السابقة لمتغير ما تساعد في التنبؤ بالقيم المستقبلية لمتغير آخر.
    """)

	st.latex(r'''
    H_0: \text{المتغير } X \text{ لا يسبب بالمعنى الجرانجري المتغير } Y
    ''')
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<h3>تطبيق نموذج VAR على البيانات</h3>', unsafe_allow_html=True)

	st.write("""
    بناءً على نتائج اختبارات الاستقرارية، نعلم أن البيانات غير مستقرة I(1). لذلك، سنطبق نموذج VAR على الفروق الأولى للبيانات.
    """)

	# الفروق الأولى للبيانات
	diff_data = data.diff().dropna()

	# الفروق الأولى للمتغيرات غير المتكاملة تكاملاً مشتركاً
	st.subheader("تطبيق VAR على الفروق الأولى للمتغيرات (y1, y3)")
	var_result = run_var_model(diff_data[['y1', 'y3']])

	# عرض عدد فترات الإبطاء المثلى
	st.write("**اختيار عدد فترات الإبطاء المثلى:**")
	lag_order_df = pd.DataFrame({
		'AIC': var_result['lag_order'].aic,
		'BIC': var_result['lag_order'].bic,
		'FPE': var_result['lag_order'].fpe,
		'HQIC': var_result['lag_order'].hqic
	}, index=[f"p={i}" for i in range(1, len(var_result['lag_order'].aic) + 1)])
	st.dataframe(lag_order_df)

	st.write(f"**عدد فترات الإبطاء المثلى حسب معيار AIC: {var_result['lag_order'].aic}**")

	# عرض ملخص النموذج
	st.write("**ملخص نموذج VAR:**")
	st.text(var_result['summary'])

	# دالة الاستجابة النبضية
	st.subheader("دالة الاستجابة النبضية")
	st.write("""
    تظهر دالة الاستجابة النبضية كيف تستجيب المتغيرات في النظام لصدمة بحجم انحراف معياري واحد في متغير آخر.
    """)

	irf_figs = plot_irf(var_result['irf'], ['y1', 'y3'])
	for fig in irf_figs:
		st.plotly_chart(fig, use_container_width=True)

	# تجزئة التباين
	st.subheader("تجزئة التباين")
	st.write("""
    تظهر تجزئة التباين مساهمة صدمات كل متغير في تباين التنبؤ للمتغيرات الأخرى عبر الزمن.
    """)

	fevd_figs = plot_fevd(var_result['fevd'], ['y1', 'y3'])
	for fig in fevd_figs:
		st.plotly_chart(fig, use_container_width=True)

	# التنبؤ
	st.subheader("التنبؤ باستخدام نموذج VAR")
	st.write("""
    يمكن استخدام نموذج VAR للتنبؤ بالقيم المستقبلية للمتغيرات.
    """)

	forecast_df = pd.DataFrame(var_result['forecast'], columns=['y1', 'y3'])
	forecast_df.index = range(len(diff_data), len(diff_data) + len(forecast_df))

	fig = go.Figure()
	# إضافة البيانات الأصلية
	for col in diff_data[['y1', 'y3']].columns:
		fig.add_trace(go.Scatter(
			x=diff_data.index,
			y=diff_data[col],
			mode='lines',
			name=f'{col} (الفعلية)'
		))

	# إضافة التنبؤات
	for col in forecast_df.columns:
		fig.add_trace(go.Scatter(
			x=forecast_df.index,
			y=forecast_df[col],
			mode='lines',
			line=dict(dash='dash'),
			name=f'{col} (التنبؤ)'
		))

	fig.update_layout(
		title="التنبؤ باستخدام نموذج VAR",
		xaxis_title="الزمن",
		yaxis_title="القيمة",
		height=500,
		template="plotly_white"
	)

	st.plotly_chart(fig, use_container_width=True)

	st.markdown('<div class="application">', unsafe_allow_html=True)
	st.write("""
    **الاستنتاج من نموذج VAR:**

    1. **عدد فترات الإبطاء المثلى**: تم اختيار عدد فترات الإبطاء بناءً على معيار AIC.

    2. **دالة الاستجابة النبضية**: تظهر كيف تستجيب المتغيرات للصدمات في النظام. نلاحظ أن:
       - صدمة في y1 لها تأثير على y1 نفسها ولكن تأثيرها على y3 ضعيف.
       - صدمة في y3 لها تأثير على y3 نفسها ولكن تأثيرها على y1 ضعيف.
       - هذا يتفق مع طريقة توليد البيانات، حيث أن y1 و y3 مستقلتان.

    3. **تجزئة التباين**: تظهر أن معظم تباين كل متغير يفسر بواسطة صدمات المتغير نفسه، وهذا يؤكد استقلالية المتغيرين.

    4. **التنبؤ**: يمكن استخدام نموذج VAR للتنبؤ بالقيم المستقبلية للمتغيرات.
    """)
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<div class="caution">', unsafe_allow_html=True)
	st.write("""
    **ملاحظة مهمة حول تطبيق نموذج VAR:**

    - استخدمنا نموذج VAR على الفروق الأولى للمتغيرات y1 و y3 لأنهما غير متكاملتين تكاملاً مشتركاً.

    - بالنسبة للمتغيرات y1 و y2، التي أظهرت اختبارات التكامل المشترك وجود علاقة تكامل مشترك بينهما، فإن نموذج VECM هو الأنسب.

    - استخدام نموذج VAR مع متغيرات I(1) متكاملة تكاملاً مشتركاً قد يؤدي إلى خسارة معلومات العلاقة طويلة الأجل.
    """)
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<h3>متى يستخدم نموذج VAR؟</h3>', unsafe_allow_html=True)
	st.markdown('<div class="note">', unsafe_allow_html=True)
	st.write("""
    **الحالات المناسبة لاستخدام نموذج VAR:**

    1. **المتغيرات مستقرة I(0)**: يمكن تطبيق نموذج VAR مباشرة على المستويات.

    2. **المتغيرات متكاملة من الدرجة الأولى I(1) وغير متكاملة تكاملاً مشتركاً**: يتم تطبيق نموذج VAR على الفروق الأولى.

    3. **قياس التأثيرات الديناميكية**: عندما نرغب في دراسة العلاقات الديناميكية قصيرة الأجل بين المتغيرات.

    4. **التنبؤ**: عندما يكون الهدف الأساسي هو التنبؤ بالقيم المستقبلية للمتغيرات.

    **الحالات غير المناسبة:**

    1. **المتغيرات متكاملة من الدرجة الأولى I(1) ومتكاملة تكاملاً مشتركاً**: في هذه الحالة، يفضل استخدام نموذج VECM.

    2. **عند الاهتمام بالعلاقات طويلة الأجل**: نموذج VAR لا يلتقط العلاقات طويلة الأجل بين المتغيرات المتكاملة تكاملاً مشتركاً.
    """)
	st.markdown('</div>', unsafe_allow_html=True)
	st.markdown('</div>', unsafe_allow_html=True)

# 6. نموذج VECM
elif menu == "نموذج VECM":
	st.markdown('<div class="rtl">', unsafe_allow_html=True)
	st.markdown('<h2 class="sub-header">نموذج تصحيح الخطأ المتجهي (Vector Error Correction Model - VECM)</h2>',
				unsafe_allow_html=True)

	st.markdown('<div class="concept">', unsafe_allow_html=True)
	st.write("""
    نموذج تصحيح الخطأ المتجهي (VECM) هو امتداد لنموذج VAR يستخدم عندما تكون المتغيرات متكاملة من الدرجة الأولى I(1) ومتكاملة تكاملاً مشتركاً. يتميز النموذج بأنه يدمج ديناميكيات المدى القصير مع العلاقة التوازنية طويلة الأجل.
    """)
	st.markdown('</div>', unsafe_allow_html=True)

	# توضيح رسومي لنموذج VECM
	st.plotly_chart(plot_vecm_illustration(), use_container_width=True)

	st.markdown('<h3>الصيغة الرياضية لنموذج VECM</h3>', unsafe_allow_html=True)
	st.markdown('<div class="formula-box">', unsafe_allow_html=True)
	st.write("نموذج VECM يمكن كتابته كالتالي:")

	st.latex(r'\Delta Y_t = \Pi Y_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta Y_{t-i} + \mu + \varepsilon_t')

	st.write("حيث:")
	st.latex(r'''
    \begin{align}
    &\Delta Y_t \text{ متجه الفروق الأولى للمتغيرات} \\
    &\Pi = \alpha \beta^\prime \text{ مصفوفة تحتوي على معلومات العلاقة طويلة الأجل} \\
    &\beta \text{ مصفوفة متجهات التكامل المشترك} \\
    &\alpha \text{ مصفوفة معاملات تصحيح الخطأ (معاملات التعديل)} \\
    &\Gamma_i \text{ مصفوفات المعاملات للديناميكيات قصيرة الأجل} \\
    &\mu \text{ متجه الثوابت} \\
    &\varepsilon_t \text{ متجه الأخطاء العشوائية}
    \end{align}
    ''')
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<h3>تفسير مكونات نموذج VECM</h3>', unsafe_allow_html=True)
	st.write("""
    **1. متجهات التكامل المشترك (β):**

    تمثل العلاقات التوازنية طويلة الأجل بين المتغيرات. على سبيل المثال، للمتغيرين y1 و y2:
    """)

	st.latex(r'\beta^\prime Y_{t-1} = \beta_1 y_{1,t-1} + \beta_2 y_{2,t-1} = 0 \text{ (في التوازن)}')

	st.write("""
    **2. معاملات تصحيح الخطأ (α):**

    تقيس سرعة التعديل نحو التوازن طويل الأجل. قيمة سالبة ومعنوية تعني أن المتغير يعود إلى التوازن بعد الانحراف عنه.
    """)

	st.latex(r'\alpha_i \text{ تفسر كنسبة الانحراف التي يتم تصحيحها في كل فترة زمنية}')

	st.write("""
    **3. معاملات الديناميكيات قصيرة الأجل (Γ):**

    تمثل تأثيرات التغيرات قصيرة الأجل في المتغيرات على بعضها البعض.
    """)

	st.markdown('<h3>خطوات تقدير نموذج VECM</h3>', unsafe_allow_html=True)
	st.write("""
    1. **اختبار استقرارية المتغيرات**: التأكد من أن المتغيرات متكاملة من الدرجة الأولى I(1).

    2. **اختبار التكامل المشترك**: استخدام اختبار جوهانسون لتحديد وجود ورتبة التكامل المشترك.

    3. **تحديد عدد فترات الإبطاء**: اختيار عدد فترات الإبطاء المناسب للنموذج.

    4. **تقدير النموذج**: تقدير معاملات النموذج (α, β, Γ).

    5. **اختبار صحة النموذج**: التحقق من استقلالية الأخطاء وثبات التباين.

    6. **تحليل النتائج**: تفسير العلاقات طويلة وقصيرة الأجل، دوال الاستجابة النبضية، إلخ.
    """)

	st.markdown('<h3>تطبيق نموذج VECM على البيانات</h3>', unsafe_allow_html=True)

	st.write("""
    بناءً على نتائج اختبارات التكامل المشترك، وجدنا أن المتغيرين y1 و y2 متكاملان تكاملاً مشتركاً. لذلك، سنطبق نموذج VECM على هذين المتغيرين.
    """)

	# تطبيق نموذج VECM
	st.subheader("تطبيق VECM على المتغيرات (y1, y2)")

	# تحديد رتبة التكامل المشترك
	johansen_result = run_johansen(data[['y1', 'y2']])
	coint_rank = johansen_result['Cointegration Rank (Trace)']

	st.write(f"رتبة التكامل المشترك من اختبار جوهانسون: {coint_rank}")

	# تقدير نموذج VECM
	vecm_result = run_vecm_model(data[['y1', 'y2']], k_ar_diff=2, coint_rank=coint_rank)

	if isinstance(vecm_result, str):
		st.warning(vecm_result)
	else:
		# عرض ملخص النموذج
		st.write("**ملخص نموذج VECM:**")
		st.text(vecm_result['summary'])

		# عرض متجه التكامل المشترك
		st.write("**متجه التكامل المشترك (β):**")
		st.write(vecm_result['beta'])

		st.write("""
        يمكن كتابة العلاقة التوازنية طويلة الأجل كالتالي:
        """)

		beta_values = vecm_result['beta']
		st.latex(f"{beta_values[0, 0]:.4f} \cdot y1 + {beta_values[1, 0]:.4f} \cdot y2 = 0")

		# عرض معاملات تصحيح الخطأ
		st.write("**معاملات تصحيح الخطأ (α):**")
		st.write(vecm_result['alpha'])

		st.write("""
        معاملات تصحيح الخطأ تفسر سرعة العودة إلى التوازن طويل الأجل:
        """)

		alpha_values = vecm_result['alpha']
		st.write(
			f"- α₁ = {alpha_values[0, 0]:.4f}: يشير إلى أن {abs(alpha_values[0, 0] * 100):.2f}% من الانحراف عن التوازن في y1 يتم تصحيحه في كل فترة زمنية.")
		st.write(
			f"- α₂ = {alpha_values[1, 0]:.4f}: يشير إلى أن {abs(alpha_values[1, 0] * 100):.2f}% من الانحراف عن التوازن في y2 يتم تصحيحه في كل فترة زمنية.")

	st.markdown('<div class="application">', unsafe_allow_html=True)
	st.write("""
    **تفسير نتائج نموذج VECM:**

    1. **متجه التكامل المشترك (β)**: يحدد العلاقة التوازنية طويلة الأجل بين y1 و y2. القيم تقريباً تعكس العلاقة الحقيقية التي تم توليد البيانات بها (y2 = 0.5 * y1 + ε).

    2. **معاملات تصحيح الخطأ (α)**:
       - معامل تصحيح الخطأ للمتغير y1 (α₁): إذا كان معنوياً وسالباً، فهذا يعني أن y1 يتكيف للعودة إلى التوازن طويل الأجل.
       - معامل تصحيح الخطأ للمتغير y2 (α₂): إذا كان معنوياً وسالباً، فهذا يعني أن y2 يتكيف للعودة إلى التوازن طويل الأجل.

    3. **الديناميكيات قصيرة الأجل**: تظهر في معاملات الفروق المبطأة للمتغيرات في النموذج.
    """)
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<h3>مقارنة بين نموذج VAR ونموذج VECM</h3>', unsafe_allow_html=True)
	col1, col2 = st.columns(2)

	with col1:
		st.markdown('<div class="application">', unsafe_allow_html=True)
		st.write("**خصائص نموذج VAR:**")
		st.write("""
        - مناسب للمتغيرات المستقرة I(0)
        - مناسب للمتغيرات I(1) غير المتكاملة تكاملاً مشتركاً (بعد أخذ الفروق)
        - يركز على الديناميكيات قصيرة الأجل
        - يفترض عدم وجود علاقات توازنية طويلة الأجل
        - أبسط في التقدير والتفسير
        """)
		st.markdown('</div>', unsafe_allow_html=True)

	with col2:
		st.markdown('<div class="application">', unsafe_allow_html=True)
		st.write("**خصائص نموذج VECM:**")
		st.write("""
        - مناسب للمتغيرات I(1) المتكاملة تكاملاً مشتركاً
        - يدمج الديناميكيات قصيرة الأجل مع العلاقة التوازنية طويلة الأجل
        - يحتوي على آلية تصحيح الخطأ للعودة إلى التوازن
        - يحتفظ بمعلومات المستويات الأصلية للمتغيرات
        - أكثر تعقيداً في التقدير والتفسير
        """)
		st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<div class="note">', unsafe_allow_html=True)
	st.write("""
    **متى تستخدم نموذج VECM؟**

    1. **عندما تكون المتغيرات متكاملة من الدرجة الأولى I(1) ومتكاملة تكاملاً مشتركاً**.

    2. **عندما تكون هناك أهمية لدراسة العلاقة التوازنية طويلة الأجل** بين المتغيرات، إضافة إلى التفاعلات قصيرة الأجل.

    3. **عندما نرغب في فهم آلية التعديل** التي تعيد المتغيرات إلى توازنها طويل الأجل بعد الصدمات.

    4. **في التحليل الاقتصادي** حيث تكون العلاقات التوازنية ذات أهمية نظرية.
    """)
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<div class="caution">', unsafe_allow_html=True)
	st.write("""
    **ملاحظات مهمة حول نموذج VECM:**

    1. **تحديد رتبة التكامل المشترك**: يجب تحديدها بشكل صحيح باستخدام اختبار جوهانسون.

    2. **تحديد عدد فترات الإبطاء**: يؤثر على نتائج النموذج ويجب اختياره بعناية.

    3. **تفسير النتائج**: يتطلب فهماً للمفاهيم الاقتصادية والإحصائية المرتبطة بالنموذج.

    4. **حجم العينة**: يحتاج نموذج VECM إلى عينة كبيرة نسبياً للحصول على تقديرات موثوقة.
    """)
	st.markdown('</div>', unsafe_allow_html=True)
	st.markdown('</div>', unsafe_allow_html=True)

# 7. السببية لجرانجر
elif menu == "السببية لجرانجر":
	st.markdown('<div class="rtl">', unsafe_allow_html=True)
	st.markdown('<h2 class="sub-header">اختبار السببية لجرانجر (Granger Causality)</h2>', unsafe_allow_html=True)

	st.markdown('<div class="concept">', unsafe_allow_html=True)
	st.write("""
    السببية لجرانجر هي مفهوم إحصائي يستخدم لتحديد ما إذا كانت سلسلة زمنية واحدة يمكن أن تساعد في التنبؤ بسلسلة زمنية أخرى. تعتمد على فكرة أن السبب يجب أن يسبق النتيجة زمنياً، وأن السبب يحتوي على معلومات مفيدة حول النتيجة لا تتوفر في أي مكان آخر.
    """)
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<h3>التعريف الرسمي للسببية لجرانجر</h3>', unsafe_allow_html=True)
	st.markdown('<div class="formula-box">', unsafe_allow_html=True)
	st.write("""
    نقول أن المتغير X يسبب بالمعنى الجرانجري المتغير Y إذا كانت القيم السابقة لـ X تساعد في التنبؤ بالقيم الحالية لـ Y بشكل أفضل من استخدام القيم السابقة لـ Y فقط.
    """)

	st.write("**الفرضية الصفرية للاختبار:**")
	st.latex(r'H_0: \text{المتغير } X \text{ لا يسبب بالمعنى الجرانجري المتغير } Y')

	st.write("**النموذج المقيد (تحت الفرضية الصفرية):**")
	st.latex(r'Y_t = \alpha_0 + \alpha_1 Y_{t-1} + \alpha_2 Y_{t-2} + \ldots + \alpha_p Y_{t-p} + \varepsilon_t')

	st.write("**النموذج غير المقيد:**")
	st.latex(
		r'Y_t = \alpha_0 + \alpha_1 Y_{t-1} + \ldots + \alpha_p Y_{t-p} + \beta_1 X_{t-1} + \ldots + \beta_p X_{t-p} + \varepsilon_t')

	st.write("**إحصائية الاختبار:**")
	st.latex(r'F = \frac{(RSS_R - RSS_{UR})/p}{RSS_{UR}/(T-2p-1)}')

	st.write("""
    حيث:
    - RSS_R: مجموع مربعات البواقي للنموذج المقيد
    - RSS_UR: مجموع مربعات البواقي للنموذج غير المقيد
    - p: عدد القيود (عدد فترات الإبطاء)
    - T: عدد المشاهدات
    """)
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<h3>أنواع العلاقات السببية</h3>', unsafe_allow_html=True)
	st.write("""
    1. **السببية أحادية الاتجاه**: X تسبب Y، ولكن Y لا تسبب X.

    2. **السببية ثنائية الاتجاه (تغذية راجعة)**: X تسبب Y، و Y تسبب X.

    3. **الاستقلال**: X لا تسبب Y، و Y لا تسبب X.

    4. **العلاقة الظاهرية**: علاقة سببية تظهر بين متغيرين بسبب تأثير متغير ثالث على كليهما.
    """)

	st.markdown('<div class="caution">', unsafe_allow_html=True)
	st.write("""
    **ملاحظات مهمة حول السببية لجرانجر:**

    1. **ليست سببية بالمعنى الفلسفي**: السببية لجرانجر لا تعني بالضرورة وجود علاقة سببية حقيقية، بل تشير فقط إلى أن متغيراً ما يساعد في التنبؤ بمتغير آخر.

    2. **متغيرات محذوفة**: وجود متغير ثالث غير مدرج في النموذج قد يؤدي إلى علاقات سببية زائفة.

    3. **الاعتماد على فترات الإبطاء**: اختيار عدد فترات الإبطاء المناسب مهم للحصول على نتائج دقيقة.

    4. **استقرارية المتغيرات**: يجب أن تكون المتغيرات مستقرة قبل إجراء الاختبار (أو استخدام الفروق الأولى للمتغيرات I(1)).
    """)
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<h3>تطبيق اختبار السببية لجرانجر على البيانات</h3>', unsafe_allow_html=True)
	st.write("""
    سنطبق اختبار السببية لجرانجر على الفروق الأولى للبيانات (حيث إن المتغيرات الأصلية I(1)).
    """)

	# الفروق الأولى للبيانات
	diff_data = data.diff().dropna()

	# اختبار السببية لجرانجر لجميع أزواج المتغيرات
	st.subheader("نتائج اختبار السببية لجرانجر")
	granger_results = run_granger_causality(diff_data, maxlag=5)

	for relation, result in granger_results.items():
		st.write(f"**{relation}:**")
		if isinstance(result, dict):
			st.write(f"أدنى قيمة p: {result['Min p-value']:.4f}")
			if result['Significant']:
				st.success(f"يوجد سببية لجرانجر (نرفض الفرضية الصفرية)")
			else:
				st.error(f"لا يوجد سببية لجرانجر (لا نستطيع رفض الفرضية الصفرية)")
		else:
			st.warning(result)

	st.markdown('<div class="application">', unsafe_allow_html=True)
	st.write("""
    **تفسير نتائج اختبار السببية لجرانجر:**

    بناءً على النتائج، يمكننا تلخيص العلاقات السببية بين المتغيرات:

    1. **العلاقة بين y1 و y2**:
       - هل y1 تسبب y2؟ (نعم/لا)
       - هل y2 تسبب y1؟ (نعم/لا)
       - نوع العلاقة: (أحادية الاتجاه / ثنائية الاتجاه / استقلال)

    2. **العلاقة بين y1 و y3**:
       - هل y1 تسبب y3؟ (نعم/لا)
       - هل y3 تسبب y1؟ (نعم/لا)
       - نوع العلاقة: (أحادية الاتجاه / ثنائية الاتجاه / استقلال)

    3. **العلاقة بين y2 و y3**:
       - هل y2 تسبب y3؟ (نعم/لا)
       - هل y3 تسبب y2؟ (نعم/لا)
       - نوع العلاقة: (أحادية الاتجاه / ثنائية الاتجاه / استقلال)

    هذه النتائج تتسق مع طريقة توليد البيانات، حيث تم إنشاء y1 و y2 بحيث يكون بينهما علاقة، بينما y3 مستقلة عن المتغيرات الأخرى.
    """)
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<h3>العلاقة بين التكامل المشترك والسببية لجرانجر</h3>', unsafe_allow_html=True)
	st.markdown('<div class="note">', unsafe_allow_html=True)
	st.write("""
    **نظرية التمثيل لجرانجر (Granger Representation Theorem):**

    تربط هذه النظرية بين مفهومي التكامل المشترك والسببية لجرانجر:

    1. **وجود تكامل مشترك يعني وجود سببية**: إذا كان هناك تكامل مشترك بين متغيرين، فيجب أن تكون هناك سببية في اتجاه واحد على الأقل.

    2. **آلية تصحيح الخطأ**: في نموذج VECM، معامل تصحيح الخطأ المعنوي يشير إلى وجود سببية طويلة الأجل.

    3. **السببية قصيرة الأجل vs طويلة الأجل**: السببية لجرانجر تقيس العلاقات قصيرة الأجل، بينما التكامل المشترك يقيس العلاقات طويلة الأجل.
    """)
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<h3>متى نستخدم اختبار السببية لجرانجر؟</h3>', unsafe_allow_html=True)
	st.write("""
    1. **لدراسة اتجاه العلاقة بين المتغيرات**: هل المتغير X يؤثر على المتغير Y، أم العكس، أم كلاهما يؤثر على الآخر؟

    2. **في تحليل السلاسل الزمنية متعددة المتغيرات**: كجزء من تحليل نماذج VAR و VECM.

    3. **في دراسات الاقتصاد القياسي**: لفحص العلاقات السببية بين المتغيرات الاقتصادية، مثل:
       - هل النمو الاقتصادي يسبب زيادة الاستثمار، أم أن الاستثمار يسبب النمو الاقتصادي؟
       - هل زيادة عرض النقود تسبب التضخم، أم أن التضخم يسبب زيادة عرض النقود؟
       - هل ارتفاع أسعار النفط يسبب تباطؤ النمو الاقتصادي؟
    """)

	st.markdown('<div class="caution">', unsafe_allow_html=True)
	st.write("""
    **الاحتياطات عند استخدام اختبار السببية لجرانجر:**

    1. **التأكد من استقرارية المتغيرات**: استخدام المتغيرات المستقرة أو الفروق الأولى للمتغيرات I(1).

    2. **اختبار التكامل المشترك**: إذا كانت المتغيرات متكاملة تكاملاً مشتركاً، فيجب استخدام نموذج VECM لاختبار السببية.

    3. **اختيار عدد فترات الإبطاء المناسب**: استخدام معايير المعلومات (AIC, BIC) لاختيار عدد الفترات.

    4. **الحذر في التفسير**: عدم الخلط بين السببية الإحصائية والسببية الحقيقية.

    5. **الأخذ في الاعتبار المتغيرات المحذوفة**: قد تكون هناك متغيرات أخرى غير مدرجة في النموذج تؤثر على العلاقة.
    """)
	st.markdown('</div>', unsafe_allow_html=True)
	st.markdown('</div>', unsafe_allow_html=True)

# 8. تطبيق عملي
elif menu == "تطبيق عملي":
	st.markdown('<div class="rtl">', unsafe_allow_html=True)
	st.markdown('<h2 class="sub-header">تطبيق عملي: تحليل متكامل باستخدام البيانات الحقيقية</h2>',
				unsafe_allow_html=True)

	st.write("""
    في هذا القسم، سنقوم بتطبيق ما تعلمناه على بيانات قد تكون من مصادر حقيقية. يمكن للمستخدم تحميل بياناته الخاصة أو استخدام البيانات النموذجية التي توفرها التطبيق.
    """)

	# خيار تحميل البيانات
	data_option = st.radio(
		"اختر مصدر البيانات:",
		["استخدام البيانات النموذجية", "تحميل بيانات خاصة"]
	)

	if data_option == "تحميل بيانات خاصة":
		uploaded_file = st.file_uploader("قم بتحميل ملف CSV أو Excel", type=["csv", "xlsx", "xls"])
		if uploaded_file is not None:
			try:
				if uploaded_file.name.endswith('.csv'):
					user_data = pd.read_csv(uploaded_file)
				else:
					user_data = pd.read_excel(uploaded_file)

				st.success("تم تحميل البيانات بنجاح!")
				st.dataframe(user_data.head())

				# اختيار المتغيرات
				st.subheader("اختيار المتغيرات للتحليل")
				selected_vars = st.multiselect(
					"اختر المتغيرات التي تريد تحليلها (2-5 متغيرات):",
					options=user_data.columns.tolist(),
					default=user_data.columns.tolist()[:3]
				)

				if len(selected_vars) < 2:
					st.warning("يرجى اختيار متغيرين على الأقل.")
				elif len(selected_vars) > 5:
					st.warning("يرجى اختيار 5 متغيرات كحد أقصى.")
				else:
					analysis_data = user_data[selected_vars]
					st.success(f"تم اختيار {len(selected_vars)} متغيرات للتحليل.")
			except Exception as e:
				st.error(f"حدث خطأ أثناء قراءة الملف: {e}")
		else:
			st.info("يرجى تحميل ملف للمتابعة.")
			analysis_data = None
	else:
		# استخدام البيانات النموذجية
		st.write("سنستخدم بيانات نموذجية تمثل متغيرات اقتصادية افتراضية.")
		# توليد بيانات أكثر واقعية
		np.random.seed(123)
		T = 200

		# متغير مشترك (اتجاه عام) - يمثل نمو الاقتصاد مثلًا
		common_trend = np.cumsum(0.5 + 0.01 * np.arange(T) + np.random.normal(0, 0.5, T))

		# متغيرات اقتصادية مختلفة
		gdp = common_trend + np.random.normal(0, 1, T)  # الناتج المحلي الإجمالي
		consumption = 0.7 * gdp + np.random.normal(0, 2, T)  # الاستهلاك
		investment = 0.2 * gdp + np.random.normal(0, 3, T)  # الاستثمار
		unemployment = 100 - 0.02 * gdp + np.random.normal(0, 0.5, T)  # البطالة
		inflation = 0.1 * np.diff(gdp, prepend=0) + np.random.normal(0, 0.3, T)  # التضخم

		# إنشاء DataFrame
		eco_data = pd.DataFrame({
			'GDP': gdp,
			'Consumption': consumption,
			'Investment': investment,
			'Unemployment': unemployment,
			'Inflation': inflation
		})

		# تسمية المتغيرات بالعربية
		eco_data_ar = pd.DataFrame({
			'الناتج المحلي الإجمالي': gdp,
			'الاستهلاك': consumption,
			'الاستثمار': investment,
			'البطالة': unemployment,
			'التضخم': inflation
		})

		st.dataframe(eco_data_ar.head())

		# اختيار المتغيرات
		st.subheader("اختيار المتغيرات للتحليل")
		selected_vars_ar = st.multiselect(
			"اختر المتغيرات التي تريد تحليلها (2-5 متغيرات):",
			options=eco_data_ar.columns.tolist(),
			default=eco_data_ar.columns.tolist()[:3]
		)

		# ترجمة الأسماء العربية إلى الإنجليزية للتحليل
		var_map = {
			'الناتج المحلي الإجمالي': 'GDP',
			'الاستهلاك': 'Consumption',
			'الاستثمار': 'Investment',
			'البطالة': 'Unemployment',
			'التضخم': 'Inflation'
		}

		selected_vars = [var_map[var] for var in selected_vars_ar]

		if len(selected_vars) < 2:
			st.warning("يرجى اختيار متغيرين على الأقل.")
			analysis_data = None
		else:
			analysis_data = eco_data[selected_vars]
			st.success(f"تم اختيار {len(selected_vars)} متغيرات للتحليل.")

	# إجراء التحليل إذا تم اختيار البيانات
	if analysis_data is not None:
		st.markdown('<h3>خطة التحليل</h3>', unsafe_allow_html=True)
		st.write("""
        سنتبع الخطوات التالية في تحليلنا:

        1. **تصور البيانات**: رسم السلاسل الزمنية والعلاقات بين المتغيرات.

        2. **اختبار استقرارية المتغيرات**: باستخدام اختبار ADF.

        3. **اختبار التكامل المشترك**: باستخدام اختبار جوهانسون.

        4. **نمذجة العلاقات**: باستخدام نموذج VAR أو VECM حسب نتائج الاختبارات السابقة.

        5. **تحليل السببية**: باستخدام اختبار السببية لجرانجر.

        6. **التنبؤ والتحليل**: دوال الاستجابة النبضية، تجزئة التباين، إلخ.
        """)

		# بدء التحليل
		st.markdown('<hr>', unsafe_allow_html=True)
		st.markdown('<h3>1. تصور البيانات</h3>', unsafe_allow_html=True)

		# رسم السلاسل الزمنية
		st.subheader("رسم السلاسل الزمنية")
		st.plotly_chart(plot_time_series(analysis_data), use_container_width=True)

		# رسم العلاقات بين المتغيرات
		st.subheader("العلاقات بين المتغيرات")
		st.plotly_chart(plot_relationships(analysis_data), use_container_width=True)

		st.markdown('<hr>', unsafe_allow_html=True)
		st.markdown('<h3>2. اختبار استقرارية المتغيرات</h3>', unsafe_allow_html=True)

		# اختبار ADF لكل متغير
		adf_results = {}
		for col in analysis_data.columns:
			adf_results[col] = run_adf_test(analysis_data[col])

		# عرض نتائج اختبار ADF
		st.subheader("نتائج اختبار جذر الوحدة (ADF)")

		adf_summary = {
			'المتغير': [],
			'إحصائية الاختبار': [],
			'قيمة p': [],
			'القيمة الحرجة (5%)': [],
			'الاستقرارية': []
		}

		for col, result in adf_results.items():
			adf_summary['المتغير'].append(col)
			adf_summary['إحصائية الاختبار'].append(result['Test Statistic'])
			adf_summary['قيمة p'].append(result['p-value'])
			adf_summary['القيمة الحرجة (5%)'].append(result['Critical Values']['5%'])
			adf_summary['الاستقرارية'].append('مستقر' if result['Stationary'] else 'غير مستقر')

		adf_df = pd.DataFrame(adf_summary)
		st.dataframe(adf_df)

		# تحديد ما إذا كانت المتغيرات مستقرة أو غير مستقرة
		stationary_vars = [col for col, result in adf_results.items() if result['Stationary']]
		non_stationary_vars = [col for col, result in adf_results.items() if not result['Stationary']]

		if len(stationary_vars) > 0:
			st.success(f"المتغيرات المستقرة I(0): {', '.join(stationary_vars)}")

		if len(non_stationary_vars) > 0:
			st.warning(f"المتغيرات غير المستقرة: {', '.join(non_stationary_vars)}")

			# اختبار ADF على الفروق الأولى للمتغيرات غير المستقرة
			st.subheader("اختبار جذر الوحدة على الفروق الأولى")

			diff_data = analysis_data[non_stationary_vars].diff().dropna()
			diff_adf_results = {}

			for col in diff_data.columns:
				diff_adf_results[col] = run_adf_test(diff_data[col])

			diff_adf_summary = {
				'المتغير': [],
				'إحصائية الاختبار': [],
				'قيمة p': [],
				'القيمة الحرجة (5%)': [],
				'الاستقرارية': []
			}

			for col, result in diff_adf_results.items():
				diff_adf_summary['المتغير'].append(col)
				diff_adf_summary['إحصائية الاختبار'].append(result['Test Statistic'])
				diff_adf_summary['قيمة p'].append(result['p-value'])
				diff_adf_summary['القيمة الحرجة (5%)'].append(result['Critical Values']['5%'])
				diff_adf_summary['الاستقرارية'].append('مستقر' if result['Stationary'] else 'غير مستقر')

			diff_adf_df = pd.DataFrame(diff_adf_summary)
			st.dataframe(diff_adf_df)

			# تحديد المتغيرات I(1)
			i1_vars = [col for col, result in diff_adf_results.items() if result['Stationary']]
			if len(i1_vars) > 0:
				st.success(f"المتغيرات المتكاملة من الدرجة الأولى I(1): {', '.join(i1_vars)}")

			# تحديد المتغيرات غير المستقرة حتى بعد أخذ الفروق الأولى
			non_i1_vars = [col for col, result in diff_adf_results.items() if not result['Stationary']]
			if len(non_i1_vars) > 0:
				st.error(f"المتغيرات غير المستقرة حتى بعد أخذ الفروق الأولى: {', '.join(non_i1_vars)}")

		st.markdown('<hr>', unsafe_allow_html=True)
		st.markdown('<h3>3. اختبار التكامل المشترك</h3>', unsafe_allow_html=True)

		# التحقق من وجود متغيرات I(1) كافية لاختبار التكامل المشترك
		if 'i1_vars' in locals() and len(i1_vars) >= 2:
			st.subheader("اختبار جوهانسون للتكامل المشترك")

			# تطبيق اختبار جوهانسون
			johansen_result = run_johansen(analysis_data[i1_vars])

			if isinstance(johansen_result, str):
				st.warning(johansen_result)
			else:
				# عرض نتائج اختبار الأثر
				st.write("**نتائج اختبار الأثر (Trace Test):**")
				trace_results = {
					'رتبة التكامل المشترك': [f'r ≤ {i}' for i in range(len(i1_vars))],
					'إحصاءة الأثر': johansen_result['Trace Statistics'],
					'القيمة الحرجة (95%)': johansen_result['Trace Critical Values (95%)']
				}
				trace_df = pd.DataFrame(trace_results)
				st.dataframe(trace_df)

				# عرض نتائج اختبار القيمة الذاتية العظمى
				st.write("**نتائج اختبار القيمة الذاتية العظمى (Max Eigenvalue Test):**")
				max_eig_results = {
					'رتبة التكامل المشترك': [f'r = {i}' for i in range(len(i1_vars))],
					'إحصاءة القيمة الذاتية العظمى': johansen_result['Max Eigenvalue Statistics'],
					'القيمة الحرجة (95%)': johansen_result['Max Eigenvalue Critical Values (95%)']
				}
				max_eig_df = pd.DataFrame(max_eig_results)
				st.dataframe(max_eig_df)

				# عرض رتبة التكامل المشترك
				coint_rank_trace = johansen_result['Cointegration Rank (Trace)']
				coint_rank_max_eig = johansen_result['Cointegration Rank (Max Eigenvalue)']

				st.write(f"**رتبة التكامل المشترك (اختبار الأثر): {coint_rank_trace}**")
				st.write(f"**رتبة التكامل المشترك (اختبار القيمة الذاتية العظمى): {coint_rank_max_eig}**")

				if coint_rank_trace > 0:
					st.success(f"يوجد {coint_rank_trace} متجهات تكامل مشترك وفقاً لاختبار الأثر")
				else:
					st.error("لا يوجد تكامل مشترك وفقاً لاختبار الأثر")

				if coint_rank_max_eig > 0:
					st.success(f"يوجد {coint_rank_max_eig} متجهات تكامل مشترك وفقاً لاختبار القيمة الذاتية العظمى")
				else:
					st.error("لا يوجد تكامل مشترك وفقاً لاختبار القيمة الذاتية العظمى")
		else:
			st.warning("لا يوجد عدد كاف من المتغيرات I(1) لإجراء اختبار التكامل المشترك.")

		st.markdown('<hr>', unsafe_allow_html=True)
		st.markdown('<h3>4. نمذجة العلاقات</h3>', unsafe_allow_html=True)

		# تحديد النموذج المناسب بناءً على نتائج الاختبارات السابقة
		if all(adf_results[col]['Stationary'] for col in analysis_data.columns):
			# جميع المتغيرات مستقرة I(0) - نموذج VAR على المستويات
			st.subheader("نموذج VAR على المستويات")
			st.write("""
			            جميع المتغيرات مستقرة I(0)، لذا سنطبق نموذج VAR مباشرة على المستويات.
			            """)

			var_result = run_var_model(analysis_data)

			# عرض عدد فترات الإبطاء المثلى
			st.write("**اختيار عدد فترات الإبطاء المثلى:**")
			lag_order_df = pd.DataFrame({
				'AIC': var_result['lag_order'].aic,
				'BIC': var_result['lag_order'].bic,
				'FPE': var_result['lag_order'].fpe,
				'HQIC': var_result['lag_order'].hqic
			}, index=[f"p={i}" for i in range(1, len(var_result['lag_order'].aic) + 1)])
			st.dataframe(lag_order_df)

			st.write(f"**عدد فترات الإبطاء المثلى حسب معيار AIC: {var_result['lag_order'].aic}**")

			# عرض ملخص النموذج
			st.write("**ملخص نموذج VAR:**")
			st.text(var_result['summary'])

			# عرض دالة الاستجابة النبضية
			st.subheader("دالة الاستجابة النبضية")
			irf_figs = plot_irf(var_result['irf'], analysis_data.columns)
			for fig in irf_figs:
				st.plotly_chart(fig, use_container_width=True)

			# عرض تجزئة التباين
			st.subheader("تجزئة التباين")
			fevd_figs = plot_fevd(var_result['fevd'], analysis_data.columns)
			for fig in fevd_figs:
				st.plotly_chart(fig, use_container_width=True)

			# عرض التنبؤ
			st.subheader("التنبؤ باستخدام نموذج VAR")
			forecast_df = pd.DataFrame(var_result['forecast'], columns=analysis_data.columns)
			forecast_df.index = range(len(analysis_data), len(analysis_data) + len(forecast_df))

			fig = go.Figure()
			# إضافة البيانات الأصلية
			for col in analysis_data.columns:
				fig.add_trace(go.Scatter(
					x=analysis_data.index,
					y=analysis_data[col],
					mode='lines',
					name=f'{col} (الفعلية)'
				))

			# إضافة التنبؤات
			for col in forecast_df.columns:
				fig.add_trace(go.Scatter(
					x=forecast_df.index,
					y=forecast_df[col],
					mode='lines',
					line=dict(dash='dash'),
					name=f'{col} (التنبؤ)'
				))

			fig.update_layout(
				title="التنبؤ باستخدام نموذج VAR",
				xaxis_title="الزمن",
				yaxis_title="القيمة",
				height=500,
				template="plotly_white"
			)

			st.plotly_chart(fig, use_container_width=True)

		elif 'i1_vars' in locals() and len(i1_vars) >= 2 and ('coint_rank_trace' in locals() and coint_rank_trace > 0):
			# المتغيرات I(1) ومتكاملة تكاملاً مشتركاً - نموذج VECM
			st.subheader("نموذج VECM")
			st.write(f"""
			            المتغيرات {', '.join(i1_vars)} متكاملة من الدرجة الأولى I(1) ومتكاملة تكاملاً مشتركاً (رتبة التكامل المشترك = {coint_rank_trace}).
			            لذلك، سنطبق نموذج VECM.
			            """)

			# تقدير نموذج VECM
			vecm_result = run_vecm_model(analysis_data[i1_vars], k_ar_diff=2, coint_rank=coint_rank_trace)

			if isinstance(vecm_result, str):
				st.warning(vecm_result)
			else:
				# عرض ملخص النموذج
				st.write("**ملخص نموذج VECM:**")
				st.text(vecm_result['summary'])

				# عرض متجه التكامل المشترك
				st.write("**متجه التكامل المشترك (β):**")
				st.write(vecm_result['beta'])

				st.write("""
			                يمكن كتابة العلاقة التوازنية طويلة الأجل كالتالي:
			                """)

				beta_formula = " + ".join(
					[f"{vecm_result['beta'][i, 0]:.4f} \cdot {i1_vars[i]}" for i in range(len(i1_vars))])
				st.latex(f"{beta_formula} = 0")

				# عرض معاملات تصحيح الخطأ
				st.write("**معاملات تصحيح الخطأ (α):**")
				st.write(vecm_result['alpha'])

				st.write("""
			                معاملات تصحيح الخطأ تفسر سرعة العودة إلى التوازن طويل الأجل:
			                """)

				for i in range(len(i1_vars)):
					st.write(
						f"- α_{i1_vars[i]} = {vecm_result['alpha'][i, 0]:.4f}: يشير إلى أن {abs(vecm_result['alpha'][i, 0] * 100):.2f}% من الانحراف عن التوازن في {i1_vars[i]} يتم تصحيحه في كل فترة زمنية.")

		elif 'i1_vars' in locals() and len(i1_vars) >= 2:
			# المتغيرات I(1) ولكن غير متكاملة تكاملاً مشتركاً - نموذج VAR على الفروق الأولى
			st.subheader("نموذج VAR على الفروق الأولى")
			st.write(f"""
			            المتغيرات {', '.join(i1_vars)} متكاملة من الدرجة الأولى I(1) ولكنها غير متكاملة تكاملاً مشتركاً.
			            لذلك، سنطبق نموذج VAR على الفروق الأولى.
			            """)

			# الفروق الأولى للمتغيرات I(1)
			diff_i1_data = analysis_data[i1_vars].diff().dropna()

			var_result = run_var_model(diff_i1_data)

			# عرض عدد فترات الإبطاء المثلى
			st.write("**اختيار عدد فترات الإبطاء المثلى:**")
			lag_order_df = pd.DataFrame({
				'AIC': var_result['lag_order'].aic,
				'BIC': var_result['lag_order'].bic,
				'FPE': var_result['lag_order'].fpe,
				'HQIC': var_result['lag_order'].hqic
			}, index=[f"p={i}" for i in range(1, len(var_result['lag_order'].aic) + 1)])
			st.dataframe(lag_order_df)

			st.write(f"**عدد فترات الإبطاء المثلى حسب معيار AIC: {var_result['lag_order'].aic}**")

			# عرض ملخص النموذج
			st.write("**ملخص نموذج VAR:**")
			st.text(var_result['summary'])

			# عرض دالة الاستجابة النبضية
			st.subheader("دالة الاستجابة النبضية")
			irf_figs = plot_irf(var_result['irf'], diff_i1_data.columns)
			for fig in irf_figs:
				st.plotly_chart(fig, use_container_width=True)

			# عرض تجزئة التباين
			st.subheader("تجزئة التباين")
			fevd_figs = plot_fevd(var_result['fevd'], diff_i1_data.columns)
			for fig in fevd_figs:
				st.plotly_chart(fig, use_container_width=True)

			# عرض التنبؤ
			st.subheader("التنبؤ باستخدام نموذج VAR على الفروق الأولى")
			forecast_df = pd.DataFrame(var_result['forecast'], columns=diff_i1_data.columns)
			forecast_df.index = range(len(diff_i1_data), len(diff_i1_data) + len(forecast_df))

			fig = go.Figure()
			# إضافة البيانات الأصلية
			for col in diff_i1_data.columns:
				fig.add_trace(go.Scatter(
					x=diff_i1_data.index,
					y=diff_i1_data[col],
					mode='lines',
					name=f'{col} (الفعلية)'
				))

			# إضافة التنبؤات
			for col in forecast_df.columns:
				fig.add_trace(go.Scatter(
					x=forecast_df.index,
					y=forecast_df[col],
					mode='lines',
					line=dict(dash='dash'),
					name=f'{col} (التنبؤ)'
				))

			fig.update_layout(
				title="التنبؤ باستخدام نموذج VAR على الفروق الأولى",
				xaxis_title="الزمن",
				yaxis_title="القيمة",
				height=500,
				template="plotly_white"
			)

			st.plotly_chart(fig, use_container_width=True)

		else:
			st.warning("لا يمكن تحديد النموذج المناسب بناءً على نتائج الاختبارات السابقة.")

		st.markdown('<hr>', unsafe_allow_html=True)
		st.markdown('<h3>5. تحليل السببية</h3>', unsafe_allow_html=True)

		# اختبار السببية لجرانجر
		st.subheader("اختبار السببية لجرانجر")

		# تحديد البيانات المناسبة للاختبار
		if all(adf_results[col]['Stationary'] for col in analysis_data.columns):
			# إذا كانت جميع المتغيرات مستقرة، نطبق الاختبار على المستويات
			granger_data = analysis_data
			st.write("تطبيق اختبار السببية لجرانجر على المستويات:")
		elif 'i1_vars' in locals() and len(i1_vars) >= 2:
			# إذا كانت المتغيرات I(1)، نطبق الاختبار على الفروق الأولى
			granger_data = analysis_data[i1_vars].diff().dropna()
			st.write("تطبيق اختبار السببية لجرانجر على الفروق الأولى:")
		else:
			st.warning("لا يمكن إجراء اختبار السببية لجرانجر.")
			granger_data = None

		if granger_data is not None:
			granger_results = run_granger_causality(granger_data, maxlag=5)

			granger_summary = {
				'العلاقة': [],
				'أدنى قيمة p': [],
				'النتيجة': []
			}

			for relation, result in granger_results.items():
				if isinstance(result, dict):
					granger_summary['العلاقة'].append(relation)
					granger_summary['أدنى قيمة p'].append(result['Min p-value'])
					granger_summary['النتيجة'].append('يوجد سببية' if result['Significant'] else 'لا يوجد سببية')

			granger_df = pd.DataFrame(granger_summary)
			st.dataframe(granger_df)

		st.markdown('<div class="conclusion">', unsafe_allow_html=True)
		st.write("""
			        **ملخص نتائج التحليل:**

			        1. **استقرارية المتغيرات**: تحديد درجة تكامل كل متغير (I(0), I(1), إلخ).

			        2. **التكامل المشترك**: تحديد وجود ورتبة التكامل المشترك بين المتغيرات.

			        3. **النموذج المناسب**: اختيار وتقدير نموذج VAR أو VECM بناءً على نتائج الاختبارات السابقة.

			        4. **تحليل العلاقات**: دراسة العلاقات الديناميكية بين المتغيرات باستخدام دوال الاستجابة النبضية وتجزئة التباين.

			        5. **السببية**: تحديد اتجاه العلاقات السببية بين المتغيرات باستخدام اختبار السببية لجرانجر.
			        """)
		st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('</div>', unsafe_allow_html=True)

	# 9. الملخص والاستنتاجات
elif menu == "الملخص والاستنتاجات":
	st.markdown('<div class="rtl">', unsafe_allow_html=True)
	st.markdown('<h2 class="sub-header">الملخص والاستنتاجات</h2>', unsafe_allow_html=True)

	st.markdown('<div class="conclusion">', unsafe_allow_html=True)
	st.write("""
			    في هذا التطبيق، تناولنا أساليب تحليل السلاسل الزمنية متعددة المتغيرات، مع التركيز على:

			    1. **نماذج VAR و VECM**: الأدوات الأساسية لتحليل العلاقات الديناميكية بين المتغيرات.

			    2. **التكامل المشترك**: مفهوم أساسي في الاقتصاد القياسي يساعد في دراسة العلاقات طويلة الأجل.

			    3. **اختبارات انجل-جرانجر وجوهانسون**: الطرق الرئيسية للكشف عن التكامل المشترك.

			    4. **السببية لجرانجر**: أداة لدراسة اتجاه العلاقات بين المتغيرات.

			    إن فهم هذه الأدوات واستخدامها بشكل صحيح يمكن أن يساعد الباحثين والمحللين في:
			    - تحديد العلاقات طويلة وقصيرة الأجل بين المتغيرات الاقتصادية
			    - التنبؤ بالقيم المستقبلية للمتغيرات
			    - فهم آليات انتقال الصدمات بين المتغيرات
			    - تحديد اتجاه العلاقات السببية
			    - تقييم فعالية السياسات الاقتصادية
			    """)
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<h3>شجرة القرار لاختيار النموذج المناسب</h3>', unsafe_allow_html=True)

	st.markdown('<div class="concept">', unsafe_allow_html=True)
	st.write("""
			    يمكن استخدام شجرة القرار التالية لاختيار النموذج المناسب:

			    1. **اختبار استقرارية المتغيرات** (اختبار ADF):
			       - إذا كانت جميع المتغيرات I(0): استخدم نموذج VAR على المستويات
			       - إذا كانت المتغيرات I(1): انتقل إلى الخطوة 2

			    2. **اختبار التكامل المشترك** (اختبار جوهانسون):
			       - إذا كان هناك تكامل مشترك: استخدم نموذج VECM
			       - إذا لم يكن هناك تكامل مشترك: استخدم نموذج VAR على الفروق الأولى
			    """)
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<h3>جدول مقارنة بين النماذج والاختبارات</h3>', unsafe_allow_html=True)

	comparison_data = {
		'النموذج/الاختبار': ['نموذج VAR', 'نموذج VECM', 'اختبار انجل-جرانجر', 'اختبار جوهانسون',
							 'اختبار السببية لجرانجر'],
		'الاستخدام الرئيسي': [
			'تحليل العلاقات الديناميكية بين متغيرات مستقرة',
			'تحليل العلاقات طويلة وقصيرة الأجل بين متغيرات متكاملة تكاملاً مشتركاً',
			'اختبار وجود تكامل مشترك بين متغيرين',
			'اختبار وجود تكامل مشترك بين متغيرين أو أكثر',
			'اختبار العلاقات السببية بين المتغيرات'
		],
		'الشروط الأساسية': [
			'المتغيرات مستقرة I(0) أو متكاملة ولكن غير متكاملة تكاملاً مشتركاً',
			'المتغيرات متكاملة من نفس الدرجة ومتكاملة تكاملاً مشتركاً',
			'المتغيرات متكاملة من نفس الدرجة (عادة I(1))',
			'المتغيرات متكاملة من نفس الدرجة (عادة I(1))',
			'المتغيرات مستقرة'
		],
		'المزايا': [
			'بسيط، مرن، يسمح بدراسة العلاقات الديناميكية',
			'يأخذ في الاعتبار العلاقة طويلة الأجل، أكثر دقة للمتغيرات المتكاملة تكاملاً مشتركاً',
			'بسيط، سهل التنفيذ والتفسير',
			'يتعامل مع أكثر من متغيرين، يكتشف متجهات تكامل مشترك متعددة',
			'يختبر اتجاه العلاقة، يمكن استخدامه مع نماذج VAR/VECM'
		],
		'العيوب': [
			'لا يلتقط العلاقات طويلة الأجل للمتغيرات المتكاملة تكاملاً مشتركاً',
			'أكثر تعقيداً، يتطلب تحديد رتبة التكامل المشترك بدقة',
			'متغيرين فقط، حساس لاتجاه الانحدار، لا يكتشف متجهات تكامل مشترك متعددة',
			'أكثر تعقيداً حسابياً، يتطلب عينات كبيرة نسبياً',
			'لا يعني بالضرورة السببية الحقيقية، حساس لاختيار فترات الإبطاء'
		]
	}

	comparison_df = pd.DataFrame(comparison_data)
	st.dataframe(comparison_df)

	st.markdown('<h3>توصيات لتطبيق النماذج في المجالات المختلفة</h3>', unsafe_allow_html=True)

	# يتم استخدام st.expander لتقسيم المحتوى إلى أجزاء قابلة للتوسعة/الطي
	with st.expander("في مجال الاقتصاد الكلي"):
		st.write("""
			        - **دراسة العلاقة بين النمو الاقتصادي والتضخم والبطالة**: استخدام نموذج VAR/VECM لفهم العلاقات الديناميكية وآثار السياسات النقدية والمالية.

			        - **تحليل العلاقة بين أسعار الفائدة قصيرة وطويلة الأجل**: استخدام اختبارات التكامل المشترك ونموذج VECM لدراسة هيكل أسعار الفائدة.

			        - **دراسة العلاقة بين الاستهلاك والدخل**: تطبيق فرضية الدخل الدائم باستخدام التكامل المشترك.
			        """)

	with st.expander("في مجال التمويل"):
		st.write("""
			        - **تحليل كفاءة السوق وعلاقات التكامل بين الأسواق المالية**: استخدام نماذج VECM لدراسة العلاقات بين أسواق الأسهم العالمية.

			        - **دراسة العلاقة بين أسعار الأسهم والمتغيرات الاقتصادية الأساسية**: استخدام التكامل المشترك لاختبار فرضية القيمة الحالية.

			        - **نمذجة تقلبات أسعار الأصول**: دمج نماذج VAR مع نماذج GARCH متعددة المتغيرات.
			        """)

	with st.expander("في مجال التجارة الدولية"):
		st.write("""
			        - **دراسة العلاقة بين أسعار الصرف وميزان المدفوعات**: استخدام نماذج VECM لتحليل آثار J-curve.

			        - **تحليل التكامل بين الأسواق السلعية الدولية**: اختبار قانون السعر الواحد باستخدام التكامل المشترك.

			        - **دراسة العلاقة بين التجارة الدولية والنمو الاقتصادي**: استخدام اختبار السببية لجرانجر.
			        """)

	with st.expander("في مجال الطاقة والبيئة"):
		st.write("""
			        - **تحليل العلاقة بين استهلاك الطاقة والنمو الاقتصادي**: استخدام نماذج VAR/VECM واختبار السببية لجرانجر.

			        - **دراسة العلاقة بين أسعار النفط والمتغيرات الاقتصادية**: تحليل آثار صدمات أسعار النفط.

			        - **تحليل العلاقة بين انبعاثات الكربون والنمو الاقتصادي**: اختبار فرضية منحنى كوزنتس البيئي.
			        """)

	st.markdown('<h3>مراجع ومصادر إضافية للتعلم</h3>', unsafe_allow_html=True)

	st.markdown('<div class="note">', unsafe_allow_html=True)
	st.write("""
			    **الكتب المرجعية:**

			    1. Lütkepohl, H. (2005). New Introduction to Multiple Time Series Analysis. Springer.

			    2. Enders, W. (2014). Applied Econometric Time Series. Wiley.

			    3. Hamilton, J. D. (1994). Time Series Analysis. Princeton University Press.

			    4. Juselius, K. (2006). The Cointegrated VAR Model: Methodology and Applications. Oxford University Press.

			    **المقالات الأساسية:**

			    1. Engle, R. F., & Granger, C. W. J. (1987). Co-integration and Error Correction: Representation, Estimation, and Testing. Econometrica, 55(2), 251-276.

			    2. Johansen, S. (1988). Statistical Analysis of Cointegration Vectors. Journal of Economic Dynamics and Control, 12(2-3), 231-254.

			    3. Sims, C. A. (1980). Macroeconomics and Reality. Econometrica, 48(1), 1-48.

			    4. Granger, C. W. J. (1969). Investigating Causal Relations by Econometric Models and Cross-spectral Methods. Econometrica, 37(3), 424-438.

			    **موارد على الإنترنت:**

			    1. [وثائق Statsmodels](https://www.statsmodels.org/): مكتبة بايثون للإحصاء ونمذجة البيانات.

			    2. [دورات في الاقتصاد القياسي على Coursera و edX](https://www.coursera.org/courses?query=econometrics)

			    3. [دروس وأمثلة في R و Python لتحليل السلاسل الزمنية](https://otexts.com/fpp2/)
			    """)
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<div class="conclusion">', unsafe_allow_html=True)
	st.write("""
			    **خاتمة:**

			    إن تحليل السلاسل الزمنية متعددة المتغيرات باستخدام نماذج VAR و VECM واختبارات التكامل المشترك يوفر أدوات قوية لفهم العلاقات الديناميكية بين المتغيرات الاقتصادية والمالية. يتطلب تطبيق هذه الأدوات بشكل صحيح فهماً جيداً للمفاهيم الإحصائية والاقتصادية الأساسية، إلى جانب حرص في تفسير النتائج.

			    نأمل أن يكون هذا التطبيق قد قدم صورة شاملة عن هذه الأدوات وكيفية استخدامها بفعالية في التحليل الاقتصادي القياسي.
			    """)
	st.markdown('</div>', unsafe_allow_html=True)
	st.markdown('</div>', unsafe_allow_html=True)

# إظهار المؤلف والمراجع
st.sidebar.markdown("---")
st.sidebar.markdown('<div class="rtl">', unsafe_allow_html=True)
st.sidebar.info("""
			### حول التطبيق

			تم إنشاء هذا التطبيق لتوضيح مفاهيم وتقنيات تحليل السلاسل الزمنية متعددة المتغيرات، مع التركيز على نماذج VAR و VECM واختبارات التكامل المشترك.

			**الإصدار**: 1.0
			""")
st.sidebar.markdown('</div>', unsafe_allow_html=True)