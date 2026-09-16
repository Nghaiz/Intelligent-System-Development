"""Dựng Đồ thị Tri thức từ dữ liệu thật và mô hình đã huấn luyện.

Chạy:

    python -m src.kg.build_graph              # tự nạp lên Neo4j nếu .env có cấu hình
    python -m src.kg.build_graph --offline    # chỉ xuất tệp, không đụng tới mạng
    python -m src.kg.build_graph --no-wipe    # nạp thêm, không xoá dữ liệu cũ

Sáu ca mẫu được chọn từ chính hai tập dữ liệu, mỗi tầng nguy cơ và mỗi phân
khúc thị trường một ca. Xác suất và giá đều **do mô hình đã lưu trong**
``models/`` **dự báo ra**, không phải số bịa: đồ thị vì thế là phần nối tiếp
thật sự của kết quả học máy, chứ không phải một hình minh hoạ rời rạc.

Cách chọn ca có tính tái lập: trong mỗi dải, lấy quan sát nằm ở **trung vị**
của dải đó. Trung vị mô tả dải trung thực hơn giá trị cực trị, và vì thứ tự
sắp xếp là xác định nên chạy lại luôn ra đúng sáu ca ấy (yêu cầu R14).

Kết quả ghi ra:

===========================  =========================================
``outputs/graph.json``       toàn bộ sáu đồ thị, dùng cho web app Phase 6
``outputs/kg_triplets.csv``  bảng bộ ba (Chủ thể, Quan hệ, Đối tượng)
``outputs/graph.cypher``     kịch bản dán vào Neo4j Browser khi không có Aura
``outputs/kg_*.html``        đồ thị tương tác pyvis, mỗi miền một tệp
``figures/kg_*.png``         ảnh tĩnh chèn thẳng vào báo cáo LaTeX
===========================  =========================================
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import config as cfg
from .. import preprocess as prep
from .. import train as tr
from . import neo4j_client as nc
from .ontology import (
    KnowledgeGraph,
    NODE_COLOURS,
    PROPERTY_SEGMENTS,
    RISK_TIERS,
    build_medical_kg,
    build_property_kg,
)

# Phiên bản lược đồ ``graph.json``. Phase 6 và Phase 8 đọc tệp này, nên mọi
# thay đổi phá vỡ tương thích phải tăng số hiệu để bên đọc phát hiện được.
SCHEMA_VERSION = 1

# Tầng hiển thị của từng loại thực thể khi vẽ ảnh tĩnh. Đồ thị ở đây có cấu
# trúc phân lớp rõ ràng, nên bố cục nhiều tầng phản ánh đúng bản thể học và
# cho ra hình xác định — khác với bố cục lò xo vốn phụ thuộc số ngẫu nhiên.
#
# Hai miền có chuỗi suy luận khác nhau nên mỗi miền một bảng tầng riêng. Dùng
# chung một bảng thì nút Pháp lý (thuộc về bất động sản, đáng lẽ đứng cạnh nó)
# bị đẩy sang tận cột tiện ích đô thị, kéo theo một cạnh cắt ngang cả hình.
NODE_LAYERS = {
    "medical": {
        "Patient": 0,
        "Biomarker": 1,
        "Model": 2,
        "Prediction": 3,
        "Disease": 4, "RiskTier": 4,
        "Device": 5, "Nutrition": 5, "Action": 5,
        "CarePackage": 6,
        "Retailer": 7, "Service": 7,
    },
    "property": {
        "Property": 0,
        "Attribute": 1, "Legal": 1,
        "Location": 2,
        "Model": 3,
        "Prediction": 4,
        "Segment": 5,
        "LoanPackage": 6, "Infrastructure": 6,
        "Repayment": 7, "Utility": 7, "Service": 7,
        "Requirement": 8,
    },
}

# Bảng trên là **thứ tự cột khi vẽ**, mịn hơn năm tầng kiến trúc khai ở
# ``ontology.KG_LAYERS``: một tầng kiến trúc có thể trải ra vài cột để đồ thị
# không bị dồn nút. Hai bảng phải phủ cùng một tập loại thực thể — bài kiểm thử
# ``test_moi_loai_thuc_the_deu_co_tang_hien_thi`` canh đúng điều đó, vì thiếu
# một loại ở đây thì nút bị dồn hết về cột cuối mà không lỗi nào được ném.

# Tên mô hình do scikit-learn và XGBoost đặt là tên lớp lập trình. Báo cáo và
# các bảng kết quả ở Phase 3 gọi chúng bằng tên thuật toán, nên đồ thị phải
# dùng đúng cách gọi ấy thì người đọc mới đối chiếu được hai nơi.
MODEL_DISPLAY_NAMES = {
    "KNeighborsClassifier": "K-Nearest Neighbors",
    "LogisticRegression": "Logistic Regression",
    "DecisionTreeClassifier": "Decision Tree",
    "DecisionTreeRegressor": "Decision Tree",
    "RandomForestClassifier": "Random Forest",
    "RandomForestRegressor": "Random Forest",
    "XGBClassifier": "XGBoost",
    "XGBRegressor": "XGBoost",
    "LinearRegression": "Linear Regression",
    "SVR": "Support Vector Regression",
}


# ==========================================================================
#  CHUẨN HOÁ KIỂU DỮ LIỆU
# ==========================================================================

def _native(value: Any) -> Any:
    """Đổi kiểu của pandas/NumPy về kiểu Python thuần.

    Đây là ranh giới bắt buộc phải đi qua: trình điều khiển Neo4j **không**
    tuần tự hoá được ``numpy.int64``, và ``json.dumps`` cũng vậy. Ngoài ra
    ``NaN`` phải thành ``None``, vì bản thể học kiểm tra thiếu dữ liệu bằng
    ``is None`` — mà ``NaN`` thì không phải ``None``, nên nếu bỏ qua bước này
    đồ thị sẽ mọc ra những nút "Insulin = nan".
    """
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (pd.isna(value) or np.isinf(value)):
        return None
    if isinstance(value, str):
        return value.strip() or None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _row_to_dict(row: pd.Series, columns: list[str]) -> dict[str, Any]:
    """Rút một số cột của quan sát thành từ điển đã chuẩn hoá kiểu."""
    return {column: _native(row.get(column)) for column in columns}


def _estimator_name(pipeline) -> str:
    """Tên thuật toán của mô hình cuối trong pipeline, để ghi vào nút Model."""
    class_name = pipeline.steps[-1][1].__class__.__name__
    return MODEL_DISPLAY_NAMES.get(class_name, class_name)


# ==========================================================================
#  MỘT CA MẪU
# ==========================================================================

@dataclass
class Case:
    """Một ca mẫu: dữ liệu vào, dự báo của mô hình, và đồ thị tri thức sinh ra."""

    id: str
    domain: str
    label: str
    model: str
    inputs: dict[str, Any]
    prediction: dict[str, Any]
    kg: KnowledgeGraph = field(repr=False)

    def to_dict(self) -> dict:
        payload = self.kg.to_dict()
        return {
            "id": self.id,
            "domain": self.domain,
            "label": self.label,
            "model": self.model,
            "inputs": self.inputs,
            "prediction": self.prediction,
            "stats": self.kg.stats(),
            **payload,
        }


def _bands(entries: list[dict], key: str) -> list[tuple[dict, float, float]]:
    """Dải giá trị của mỗi tầng: ``[ngưỡng của chính nó, ngưỡng của tầng ngay trên)``.

    Bảng tầng xếp giảm dần, nên tầng đầu tiên mở lên vô cực còn mỗi tầng sau bị
    chặn trên bởi ngưỡng của tầng liền trước. Cách chia này khớp từng điểm một
    với ``classify_risk`` và ``classify_segment`` — điều kiện để ca được chọn
    cho một tầng thật sự thuộc về tầng ấy.

    Viết chung cho cả hai miền vì hai bảng tầng có cùng hình dạng; tách đôi thì
    chỉ cần sửa một bên là hai bên lệch nhau mà không ai phát hiện.
    """
    return [
        (entry, entry[key], entries[position - 1][key] if position > 0 else np.inf)
        for position, entry in enumerate(entries)
    ]


def _median_of_band(values: pd.Series, low: float, high: float) -> int | None:
    """Vị trí của quan sát nằm giữa dải ``[low, high)``, hoặc ``None`` nếu dải rỗng.

    Sắp xếp theo cặp (giá trị dự báo, chỉ số gốc) — một thứ tự **toàn phần**, nên
    kết quả không phụ thuộc thứ tự đầu vào kể cả khi nhiều quan sát trùng giá
    trị. Đó là tình huống thường gặp chứ không hiếm: K-Nearest Neighbors với
    k = 5 chỉ trả về sáu mức xác suất rời rạc, nên hàng trăm bệnh nhân cùng
    mang một con số. Chỉ dựa vào tính ổn định của thuật toán sắp xếp thì thứ tự
    đầu vào lại quyết định kết quả, và yêu cầu tái lập R14 sụp đổ.
    """
    band = values[(values >= low) & (values < high)]
    if band.empty:
        return None
    ordered = [index for _value, index in sorted(zip(band.to_numpy(), band.index))]
    return ordered[len(ordered) // 2]


# ==========================================================================
#  MIỀN 1 — Y SINH
# ==========================================================================

MEDICAL_FIELDS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                  "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]


def select_medical_cases() -> list[Case]:
    """Chọn một bệnh nhân thật cho mỗi tầng nguy cơ lâm sàng."""
    model = tr.load_model("diabetes_best.joblib")
    model_name = _estimator_name(model)
    X, y = prep.prepare_diabetes()
    proba = pd.Series(model.predict_proba(X)[:, 1], index=X.index)

    cases: list[Case] = []
    for tier, low, high in _bands(RISK_TIERS, "min_prob"):
        index = _median_of_band(proba, low, high)
        if index is None:
            print(f"  ! Không có bệnh nhân nào rơi vào {tier['label']}, bỏ qua ca này")
            continue

        patient = _row_to_dict(X.loc[index], MEDICAL_FIELDS)
        probability = float(proba.loc[index])
        cases.append(Case(
            id=tier["id"].replace("tier_", "med_"),
            domain="medical",
            label=f"{tier['label']} — bệnh nhân #{index}",
            model=model_name,
            inputs=patient,
            prediction={
                "Xác suất mắc bệnh": round(probability, 4),
                "Nhãn thật": int(y.loc[index]),
                "Tầng nguy cơ": tier["label"],
                "Mã ICD-10": tier["icd10"],
            },
            kg=build_medical_kg(patient, probability, model_name=model_name),
        ))
    return cases


# ==========================================================================
#  MIỀN 2 — ĐÔ THỊ VÀ TÀI CHÍNH
# ==========================================================================

PROPERTY_FIELDS = ["Area", "Frontage", "Access Road", "House direction", "Floors",
                   "Bedrooms", "Bathrooms", "Legal status", "Furniture state",
                   "Province", "District"]


def select_property_cases() -> list[Case]:
    """Chọn một bất động sản thật cho mỗi phân khúc thị trường."""
    model = tr.load_model("housing_best.joblib")
    model_name = _estimator_name(model)
    X, y, _dropped = prep.prepare_housing()
    predicted = pd.Series(model.predict(X), index=X.index)

    cases: list[Case] = []
    for segment, low, high in _bands(PROPERTY_SEGMENTS, "min_price"):
        index = _median_of_band(predicted, low, high)
        if index is None:
            print(f"  ! Không có bất động sản nào rơi vào {segment['label']}, bỏ qua")
            continue

        prop = _row_to_dict(X.loc[index], PROPERTY_FIELDS)
        price = float(predicted.loc[index])
        cases.append(Case(
            id=segment["id"].replace("seg_", "prop_"),
            domain="property",
            label=f"{segment['label']} — tin đăng #{index}",
            model=model_name,
            inputs=prop,
            prediction={
                "Giá định giá (tỷ VNĐ)": round(price, 3),
                "Giá thật (tỷ VNĐ)": round(float(y.loc[index]), 3),
                "Sai số tuyệt đối": round(abs(price - float(y.loc[index])), 3),
                "Phân khúc": segment["label"],
                "Gói vay": segment["loan"],
            },
            kg=build_property_kg(prop, price, model_name=model_name),
        ))
    return cases


# ==========================================================================
#  XUẤT TỆP
# ==========================================================================

DOMAIN_LABELS = {
    "medical": "Đồ thị Tri thức Y sinh",
    "property": "Đồ thị Tri thức Đô thị – Tài chính",
}


def export_json(cases: list[Case]) -> Path:
    """Ghi ``outputs/graph.json`` — nguồn dữ liệu chung cho Phase 6 và Phase 8."""
    document = {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "sinh_boi": "src/kg/build_graph.py",
            "so_ca": len(cases),
            "bang_mau": NODE_COLOURS,
        },
        "domains": {
            domain: {
                "label": DOMAIN_LABELS[domain],
                "cases": [c.to_dict() for c in cases if c.domain == domain],
            }
            for domain in DOMAIN_LABELS
        },
    }
    path = cfg.OUTPUTS_DIR / "graph.json"
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


def export_triplets(cases: list[Case]) -> tuple[Path, pd.DataFrame]:
    """Ghi bảng bộ ba tri thức — bằng chứng KG = (V, E, R) ở dạng đọc được."""
    rows = [
        {
            "Miền": DOMAIN_LABELS[case.domain],
            "Ca": case.label,
            "Chủ thể": subject,
            "Quan hệ": relation,
            "Đối tượng": obj,
        }
        for case in cases
        for subject, relation, obj in case.kg.triplets()
    ]
    frame = pd.DataFrame(rows)
    path = cfg.OUTPUTS_DIR / "kg_triplets.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path, frame


def export_cypher(cases: list[Case]) -> Path:
    """Ghi kịch bản Cypher rời cho người không có tài khoản Aura."""
    all_nodes: list[dict] = []
    all_edges: list[dict] = []
    for case in cases:
        nodes, edges = nc.graph_rows(case.kg, case.id, case.domain)
        all_nodes.extend(nodes)
        all_edges.extend(edges)

    path = cfg.OUTPUTS_DIR / "graph.cypher"
    path.write_text(nc.to_cypher(all_nodes, all_edges), encoding="utf-8")
    return path


def _merged_graph(cases: list[Case]):
    """Gộp nhiều ca thành một đồ thị networkx, định danh có tiền tố theo ca."""
    import networkx as nx

    graph = nx.DiGraph()
    for case in cases:
        layers = NODE_LAYERS[case.domain]
        for node in case.kg.nodes:
            graph.add_node(
                f"{case.id}::{node.id}",
                label=node.label,
                kind=node.type,
                case=case.label,
                layer=layers.get(node.type, max(layers.values())),
                properties=node.properties,
            )
        for edge in case.kg.edges:
            graph.add_edge(f"{case.id}::{edge.source}",
                           f"{case.id}::{edge.target}",
                           relation=edge.relation)
    return graph


def strip_cdn_links(html: str) -> str:
    """Gỡ các thẻ pyvis trỏ ra Internet khỏi trang HTML đã sinh.

    pyvis vẫn chèn hai đường dẫn Bootstrap trỏ ra ngoài kể cả khi đã yêu cầu
    nhúng thư viện tại chỗ bằng ``cdn_resources="in_line"``. Bootstrap chỉ tô
    điểm cho thanh công cụ lọc mà đồ thị này không bật, nên gỡ đi để trang mở
    được ngay cả khi không có mạng — điều kiện của việc nộp bài kèm tệp HTML
    rời, và cũng là điều kiện để ứng dụng web Phase 6 nhúng đồ thị dựng tại chỗ
    mà không phụ thuộc đường truyền của người dùng.

    Cả hai nơi sinh HTML — tệp ``outputs/kg_*.html`` ở đây và đồ thị dựng trực
    tiếp từ dữ liệu người dùng nhập ở ``app/lib.py`` — đều gọi hàm này. Chép
    biểu thức chính quy sang chỗ thứ hai thì sửa một bên là hai bên lệch nhau
    mà không lỗi nào được ném ra.
    """
    return re.sub(r"[ \t]*<(?:link|script)[^>]*cdn\.jsdelivr\.net[^>]*>"
                  r"(?:</script>)?[ \t]*\n?", "", html)


def export_html(cases: list[Case], domain: str) -> Path:
    """Vẽ đồ thị tương tác bằng pyvis, mỗi miền một tệp HTML.

    Dùng ``cdn_resources="in_line"`` để nhúng thẳng thư viện JavaScript vào
    tệp: HTML mở được khi không có mạng, và Phase 6 nhúng được vào Streamlit
    mà không cần phục vụ thêm tệp tĩnh nào.
    """
    from pyvis.network import Network

    subset = [c for c in cases if c.domain == domain]
    graph = _merged_graph(subset)

    net = Network(height="720px", width="100%", directed=True,
                  bgcolor="#FFFFFF", font_color="#222222",
                  cdn_resources="in_line", notebook=False)
    net.barnes_hut(gravity=-9000, spring_length=180, spring_strength=0.02)

    for node_id, data in graph.nodes(data=True):
        tooltip = [f"{data['kind']} · {data['case']}"]
        tooltip += [f"{k}: {v}" for k, v in data["properties"].items()]
        net.add_node(
            node_id,
            label=textwrap.shorten(data["label"], width=42, placeholder="…"),
            title="\n".join(tooltip),
            color=NODE_COLOURS.get(data["kind"], "#95A5A6"),
            shape="dot",
            size=22 if data["layer"] == 0 else 14,
        )
    for source, target, data in graph.edges(data=True):
        net.add_edge(source, target, label=data["relation"],
                     color="#B0B7BF", font={"size": 9, "color": "#5D6D7E"})

    path = cfg.OUTPUTS_DIR / f"kg_{domain}.html"
    # pyvis ghi tệp theo đường dẫn tương đối với thư mục làm việc; truyền chuỗi
    # tuyệt đối để kết quả không phụ thuộc chỗ đứng khi gọi lệnh.
    net.write_html(str(path), notebook=False, open_browser=False)

    path.write_text(strip_cdn_links(path.read_text(encoding="utf-8")),
                    encoding="utf-8")
    return path


def _layered_positions(graph) -> dict[str, tuple[float, float]]:
    """Xếp nút thành các cột theo tầng, trong mỗi cột gom theo loại thực thể.

    Tự tính toạ độ thay vì gọi ``multipartite_layout`` vì cần quyết định **thứ
    tự dọc** bên trong mỗi cột: để bảy nút của tầng can thiệp nằm xen kẽ nhau
    (thiết bị – dinh dưỡng – hướng dẫn – thiết bị ...) thì các cạnh đi tới
    chúng cắt nhau chằng chịt. Gom cùng loại về cạnh nhau là gỡ được phần lớn
    chỗ cắt, mà vẫn giữ tính xác định vì thứ tự sắp xếp cố định.
    """
    columns: dict[int, list[str]] = {}
    for node, data in graph.nodes(data=True):
        columns.setdefault(data["layer"], []).append(node)

    tallest = max(len(nodes) for nodes in columns.values())
    positions: dict[str, tuple[float, float]] = {}
    for layer, nodes in columns.items():
        nodes.sort(key=lambda n: (graph.nodes[n]["kind"], graph.nodes[n]["label"]))
        count = len(nodes)
        for position, node in enumerate(nodes):
            # Cột ít nút được giãn ra cho cân với cột đông nhất, thay vì dồn sát
            # vào giữa — nếu không, một cột hai nút sẽ dính liền nhau.
            spread = (count - 1) / max(tallest - 1, 1)
            offset = 0.0 if count == 1 else (position / (count - 1) - 0.5) * spread
            positions[node] = (float(layer), offset * 2.0)
    return positions


def _draw_edge_labels(graph, positions, ax) -> None:
    """Vẽ nhãn quan hệ sao cho chúng không chồng lên nhau.

    Hai biện pháp, cùng nhằm một chuyện. Thứ nhất, mỗi cặp (nút nguồn, quan hệ)
    chỉ ghi nhãn **một lần**: ba cạnh ``ADVISES`` toả ra từ cùng một tầng nguy
    cơ thì ghi ba lần là thừa. Thứ hai, các nhóm nhãn còn lại được đặt ở những
    vị trí khác nhau dọc theo cạnh, thay vì tất cả cùng nằm ở điểm giữa.

    Không làm vậy thì nhãn chồng lên nhau và matplotlib vẽ đè, cho ra những
    chuỗi ký tự vô nghĩa kiểu ``REQIADVISESJTRITION`` — ba nhãn khác nhau in
    trùng chỗ.
    """
    import networkx as nx

    groups: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for source, target, data in graph.edges(data=True):
        groups.setdefault((source, data["relation"]), []).append((source, target))

    others = np.array([positions[node] for node in graph])
    candidates = [0.28, 0.36, 0.44, 0.52, 0.60, 0.68, 0.76]

    for _key, edges in sorted(groups.items()):
        representative = edges[len(edges) // 2]
        start = np.array(positions[representative[0]])
        end = np.array(positions[representative[1]])

        # Chọn điểm đặt nhãn xa mọi nút nhất. Nhãn rơi trúng một nút thì chữ bị
        # đè, và đó là chỗ hỏng còn lại sau khi đã gỡ được chuyện nhãn chồng nhãn.
        best = max(candidates,
                   key=lambda t: float(np.min(np.linalg.norm(
                       others - (start + (end - start) * t), axis=1))))

        nx.draw_networkx_edge_labels(
            graph, positions, ax=ax, font_size=5.5, font_color="#7F8C8D",
            rotate=False, label_pos=best,
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none",
                  "alpha": 0.85},
            edge_labels={representative: graph.edges[representative]["relation"]},
        )


def export_png(case: Case, name: str, title: str) -> Path:
    """Vẽ ảnh tĩnh một ca, bố cục phân tầng theo bản thể học, để chèn vào LaTeX."""
    import matplotlib.pyplot as plt
    import networkx as nx

    cfg.apply_plot_style()
    graph = _merged_graph([case])

    positions = _layered_positions(graph)
    fig, ax = plt.subplots(figsize=(18, 11))

    colours = [NODE_COLOURS.get(graph.nodes[n]["kind"], "#95A5A6") for n in graph]
    sizes = [1500 if graph.nodes[n]["layer"] == 0 else 850 for n in graph]

    nx.draw_networkx_edges(graph, positions, ax=ax, edge_color="#CFD6DC",
                           arrows=True, arrowsize=11, width=1.0,
                           connectionstyle="arc3,rad=0.06")
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_color=colours,
                           node_size=sizes, edgecolors="white", linewidths=1.5)
    nx.draw_networkx_labels(
        graph, positions, ax=ax, font_size=6.5,
        labels={n: textwrap.fill(textwrap.shorten(graph.nodes[n]["label"],
                                                  width=44, placeholder="…"),
                                 width=20)
                for n in graph},
    )
    _draw_edge_labels(graph, positions, ax)

    layers = NODE_LAYERS[case.domain]
    kinds = sorted({graph.nodes[n]["kind"] for n in graph},
                   key=lambda k: (layers.get(k, 99), k))
    ax.legend(
        handles=[plt.Line2D([], [], marker="o", linestyle="", markersize=8,
                            markerfacecolor=NODE_COLOURS.get(k, "#95A5A6"),
                            markeredgecolor="white", label=k)
                 for k in kinds],
        loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False,
        fontsize=8, title="Loại thực thể", title_fontsize=9,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.margins(x=0.06, y=0.04)
    ax.axis("off")
    return cfg.save_fig(fig, name)


# ==========================================================================
#  NẠP LÊN NEO4J
# ==========================================================================

def push_to_neo4j(cases: list[Case], wipe: bool = True) -> dict | None:
    """Nạp toàn bộ ca lên Neo4j. Trả về ``None`` khi không kết nối được.

    Thất bại ở đây **không** làm hỏng phase: mọi tệp offline đã ghi xong trước
    khi hàm này chạy. Nhưng thất bại cũng không bị nuốt im lặng — lý do được in
    ra màn hình đầy đủ để người dùng biết đường sửa.
    """
    settings = nc.settings_from_env()
    if settings is None:
        print("  ~ Chưa cấu hình Neo4j trong .env — dùng đồ thị offline "
              "(xem setup-guides/01-neo4j-aura.md nếu muốn bật)")
        return None

    try:
        with nc.Neo4jClient(settings) as client:
            print(f"  Kết nối: {settings.describe()}")
            print(f"  Máy chủ: {client.server_version()}")
            client.ensure_constraint()
            if wipe:
                removed = client.wipe()
                print(f"  Đã xoá {removed} nút cũ để lần nạp này tái lập được")

            for case in cases:
                counts = client.load(case.kg, case_id=case.id, domain=case.domain)
                print(f"  + {case.label}: {counts['nodes']} nút, {counts['edges']} cạnh")

            report = {
                "nodes": client.count_nodes(),
                "edges": client.count_edges(),
                "by_kind": client.summary_by_kind(),
            }
            print(f"  Tổng trên máy chủ: {report['nodes']} nút, {report['edges']} cạnh")
            return report
    except nc.Neo4jUnavailable as exc:
        print(f"  ! Bỏ qua Neo4j: {exc}")
        print("  ~ Hệ thống tiếp tục chạy bằng đồ thị offline")
        return None


# ==========================================================================
#  ĐIỂM VÀO
# ==========================================================================

def build_all(offline: bool = False, wipe: bool = True) -> dict:
    """Dựng, xuất và (tuỳ chọn) nạp toàn bộ đồ thị tri thức."""
    print("=" * 74)
    print("  PHASE 5 — DỰNG ĐỒ THỊ TRI THỨC")
    print("=" * 74)

    print("\n[1/4] Chọn ca mẫu từ dữ liệu thật và dự báo bằng mô hình đã lưu")
    cases = select_medical_cases() + select_property_cases()
    for case in cases:
        stats = case.kg.stats()
        print(f"  · {case.label}: {stats['Số thực thể (|V|)']} thực thể, "
              f"{stats['Số cạnh (|E|)']} cạnh, "
              f"{stats['Số loại quan hệ (|R|)']} loại quan hệ")

    print("\n[2/4] Xuất tệp dữ liệu")
    json_path = export_json(cases)
    triplet_path, triplets = export_triplets(cases)
    cypher_path = export_cypher(cases)
    print(f"  · {json_path.name} — {len(cases)} ca, hai miền")
    print(f"  · {triplet_path.name} — {len(triplets)} bộ ba tri thức")
    print(f"  · {cypher_path.name} — kịch bản dán vào Neo4j Browser")

    print("\n[3/4] Vẽ đồ thị")
    figures = []
    for domain, richest in [("medical", "med_high"), ("property", "prop_luxury")]:
        html_path = export_html(cases, domain)
        print(f"  · {html_path.name} — đồ thị tương tác, mọi ca của miền")
        case = next((c for c in cases if c.id == richest), None)
        if case is None:
            continue
        png_path = export_png(case, f"kg_{domain}",
                              f"{DOMAIN_LABELS[domain]} — {case.label}")
        figures.append(png_path)
        print(f"  · {png_path.name} — ảnh tĩnh cho báo cáo")

    print("\n[4/4] Nạp lên Neo4j")
    report = None if offline else push_to_neo4j(cases, wipe=wipe)
    if offline:
        print("  ~ Bỏ qua theo yêu cầu (--offline)")

    print("\n" + "=" * 74)
    print(f"  HOÀN TẤT — {len(cases)} đồ thị, {len(triplets)} bộ ba, "
          f"{len(figures)} ảnh tĩnh")
    print(f"  Neo4j: {'đã nạp' if report else 'không dùng, đồ thị offline vẫn đủ'}")
    print("=" * 74)

    return {"cases": cases, "triplets": triplets, "neo4j": report}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dựng Đồ thị Tri thức cho hai miền ứng dụng.")
    parser.add_argument("--offline", action="store_true",
                        help="Chỉ xuất tệp, không kết nối Neo4j")
    parser.add_argument("--no-wipe", dest="wipe", action="store_false",
                        help="Giữ dữ liệu đang có trên Neo4j thay vì xoá đi nạp lại")
    args = parser.parse_args()
    build_all(offline=args.offline, wipe=args.wipe)


if __name__ == "__main__":
    main()
