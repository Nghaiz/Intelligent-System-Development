// Ứng dụng Flutter — Sàng lọc nguy cơ tiểu đường (Hệ thống 1)
// Gọi REST API thuần NumPy chạy tại http://10.0.2.2:5001/diabetes/v1
// (10.0.2.2 là địa chỉ máy chủ 127.0.0.1 nhìn từ trình giả lập Android)

import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const DiabetesApp());
}

const String kBaseUrl = 'http://10.0.2.2:5001/diabetes/v1';
const Color kPrimary = Color(0xFF0D6EFD);
const Color kAccent = Color(0xFFDC3545);

class DiabetesApp extends StatelessWidget {
  const DiabetesApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Sàng lọc tiểu đường',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: kPrimary,
        fontFamily: 'Roboto',
      ),
      home: const PredictPage(),
    );
  }
}

class PredictPage extends StatefulWidget {
  const PredictPage({super.key});

  @override
  State<PredictPage> createState() => _PredictPageState();
}

class _PredictPageState extends State<PredictPage> {
  final _formKey = GlobalKey<FormState>();

  String _gender = 'Female';
  String _smoking = 'never';
  final _ageCtrl = TextEditingController(text: '54');
  int _hypertension = 0;
  int _heartDisease = 0;
  final _bmiCtrl = TextEditingController(text: '27.3');
  final _hba1cCtrl = TextEditingController(text: '6.2');
  final _glucoseCtrl = TextEditingController(text: '155');

  bool _loading = false;
  String? _errorMessage;
  Map<String, dynamic>? _result;

  Map<String, dynamic>? _health;

  static const List<String> genderOptions = ['Female', 'Male'];
  static const List<String> smokingOptions = [
    'No Info',
    'current',
    'ever',
    'former',
    'never',
    'not current',
  ];
  static const Map<String, String> genderLabels = {
    'Female': 'Nữ',
    'Male': 'Nam',
  };
  static const Map<String, String> smokingLabels = {
    'No Info': 'Không có thông tin',
    'current': 'Đang hút',
    'ever': 'Đã từng hút',
    'former': 'Đã bỏ thuốc',
    'never': 'Chưa từng hút',
    'not current': 'Hiện không hút',
  };
  static const Map<String, String> tierLabels = {
    'dia_high': 'Nguy cơ cao',
    'dia_moderate': 'Nguy cơ trung bình',
    'dia_low': 'Nguy cơ thấp',
  };

  @override
  void initState() {
    super.initState();
    _loadHealth();
  }

  Future<void> _loadHealth() async {
    try {
      final resp = await http.get(Uri.parse('$kBaseUrl/health'));
      if (resp.statusCode == 200) {
        setState(() {
          _health = jsonDecode(resp.body) as Map<String, dynamic>;
        });
      }
    } catch (_) {
      // Bỏ qua — huy hiệu mô hình sẽ hiển thị trạng thái mặc định.
    }
  }

  void _fillSample() {
    setState(() {
      _gender = 'Female';
      _ageCtrl.text = '54';
      _hypertension = 0;
      _heartDisease = 0;
      _smoking = 'never';
      _bmiCtrl.text = '27.3';
      _hba1cCtrl.text = '6.2';
      _glucoseCtrl.text = '155';
    });
  }

  Future<void> _predict() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() {
      _loading = true;
      _errorMessage = null;
      _result = null;
    });

    final payload = {
      'gender': _gender,
      'age': double.tryParse(_ageCtrl.text) ?? 0,
      'hypertension': _hypertension,
      'heart_disease': _heartDisease,
      'smoking_history': _smoking,
      'bmi': double.tryParse(_bmiCtrl.text) ?? 0,
      'HbA1c_level': double.tryParse(_hba1cCtrl.text) ?? 0,
      'blood_glucose_level': double.tryParse(_glucoseCtrl.text) ?? 0,
    };

    try {
      final resp = await http.post(
        Uri.parse('$kBaseUrl/predict'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(payload),
      );
      final body = jsonDecode(resp.body) as Map<String, dynamic>;
      if (resp.statusCode == 200) {
        setState(() => _result = body);
      } else {
        setState(() => _errorMessage = body['error']?.toString() ?? 'Đã có lỗi xảy ra.');
      }
    } catch (e) {
      setState(() => _errorMessage = 'Không thể kết nối tới máy chủ (cổng 5001).');
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Sàng lọc tiểu đường'),
        backgroundColor: kPrimary,
        foregroundColor: Colors.white,
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            const Text(
              'Hệ thống 1 · 100,000 hồ sơ bệnh nhân',
              style: TextStyle(color: Colors.grey, fontSize: 13),
            ),
            const SizedBox(height: 10),
            _buildModelBadge(),
            const SizedBox(height: 16),
            _buildForm(),
            const SizedBox(height: 16),
            if (_errorMessage != null) _buildErrorBox(_errorMessage!),
            if (_loading) const Padding(
              padding: EdgeInsets.symmetric(vertical: 12),
              child: Center(child: CircularProgressIndicator()),
            ),
            if (_result != null) _buildResult(_result!),
          ],
        ),
      ),
    );
  }

  Widget _buildModelBadge() {
    final connected = _health != null;
    final graphOk = _health != null && _health!['knowledge_graph'] == 'connected';
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        border: Border.all(color: Colors.grey.shade300),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            connected
                ? '${_health!['architecture']} · ${_health!['n_params']} tham số\nSuy luận thuần NumPy — không dùng framework học sâu'
                : 'Đang kết nối máy chủ mô hình…',
            style: const TextStyle(fontSize: 12.5),
          ),
          const SizedBox(height: 6),
          Row(
            children: [
              Icon(Icons.circle, size: 8, color: graphOk ? Colors.green : Colors.grey),
              const SizedBox(width: 6),
              Text(
                graphOk ? 'Đồ thị tri thức: đã kết nối' : 'Đồ thị tri thức: không khả dụng',
                style: const TextStyle(fontSize: 11.5),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildForm() {
    return Form(
      key: _formKey,
      child: Column(
        children: [
          DropdownButtonFormField<String>(
            value: _gender,
            decoration: const InputDecoration(labelText: 'Giới tính', border: OutlineInputBorder()),
            items: genderOptions
                .map((g) => DropdownMenuItem(value: g, child: Text(genderLabels[g]!)))
                .toList(),
            onChanged: (v) => setState(() => _gender = v!),
          ),
          const SizedBox(height: 10),
          TextFormField(
            controller: _ageCtrl,
            keyboardType: TextInputType.number,
            decoration: const InputDecoration(labelText: 'Tuổi (1–120)', border: OutlineInputBorder()),
            validator: (v) {
              final n = double.tryParse(v ?? '');
              if (n == null || n < 1 || n > 120) return 'Tuổi phải trong khoảng 1–120';
              return null;
            },
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              Expanded(
                child: DropdownButtonFormField<int>(
                  value: _hypertension,
                  decoration: const InputDecoration(labelText: 'Tăng huyết áp', border: OutlineInputBorder()),
                  items: const [
                    DropdownMenuItem(value: 0, child: Text('Không')),
                    DropdownMenuItem(value: 1, child: Text('Có')),
                  ],
                  onChanged: (v) => setState(() => _hypertension = v!),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: DropdownButtonFormField<int>(
                  value: _heartDisease,
                  decoration: const InputDecoration(labelText: 'Bệnh tim', border: OutlineInputBorder()),
                  items: const [
                    DropdownMenuItem(value: 0, child: Text('Không')),
                    DropdownMenuItem(value: 1, child: Text('Có')),
                  ],
                  onChanged: (v) => setState(() => _heartDisease = v!),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          DropdownButtonFormField<String>(
            value: _smoking,
            decoration: const InputDecoration(labelText: 'Tiền sử hút thuốc', border: OutlineInputBorder()),
            items: smokingOptions
                .map((s) => DropdownMenuItem(value: s, child: Text(smokingLabels[s]!)))
                .toList(),
            onChanged: (v) => setState(() => _smoking = v!),
          ),
          const SizedBox(height: 10),
          TextFormField(
            controller: _bmiCtrl,
            keyboardType: const TextInputType.numberWithOptions(decimal: true),
            decoration: const InputDecoration(labelText: 'BMI (10–80)', border: OutlineInputBorder()),
            validator: (v) {
              final n = double.tryParse(v ?? '');
              if (n == null || n < 10 || n > 80) return 'BMI phải trong khoảng 10–80';
              return null;
            },
          ),
          const SizedBox(height: 10),
          TextFormField(
            controller: _hba1cCtrl,
            keyboardType: const TextInputType.numberWithOptions(decimal: true),
            decoration: const InputDecoration(labelText: 'HbA1c % (3–20)', border: OutlineInputBorder()),
            validator: (v) {
              final n = double.tryParse(v ?? '');
              if (n == null || n < 3 || n > 20) return 'HbA1c phải trong khoảng 3–20';
              return null;
            },
          ),
          const SizedBox(height: 10),
          TextFormField(
            controller: _glucoseCtrl,
            keyboardType: TextInputType.number,
            decoration: const InputDecoration(labelText: 'Đường huyết mg/dL (50–500)', border: OutlineInputBorder()),
            validator: (v) {
              final n = double.tryParse(v ?? '');
              if (n == null || n < 50 || n > 500) return 'Đường huyết phải trong khoảng 50–500';
              return null;
            },
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: OutlinedButton(
                  onPressed: _fillSample,
                  child: const Text('Điền dữ liệu mẫu'),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: ElevatedButton(
                  onPressed: _loading ? null : _predict,
                  style: ElevatedButton.styleFrom(backgroundColor: kPrimary, foregroundColor: Colors.white),
                  child: const Text('Dự đoán'),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildErrorBox(String message) {
    return Container(
      padding: const EdgeInsets.all(12),
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: const Color(0xFFFDECEA),
        border: Border.all(color: const Color(0xFFF5C2C7)),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(message, style: const TextStyle(color: Color(0xFF842029))),
    );
  }

  Widget _buildResult(Map<String, dynamic> data) {
    final double prob = (data['probability'] as num).toDouble();
    final double threshold = (data['threshold'] as num).toDouble();
    final int label = data['label'] as int;
    final String labelText = data['label_text'] as String;
    final String? riskTier = data['risk_tier'] as String?;
    final List knowledge = (data['knowledge'] as List?) ?? [];
    final String? knowledgeError = data['knowledge_error'] as String?;

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        border: Border.all(color: Colors.grey.shade300),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '${(prob * 100).toStringAsFixed(1)}%',
            style: const TextStyle(fontSize: 28, fontWeight: FontWeight.w800, color: kPrimary),
          ),
          Text(
            labelText,
            style: TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.bold,
              color: label == 1 ? kAccent : const Color(0xFF198754),
            ),
          ),
          const SizedBox(height: 8),
          Stack(
            children: [
              Container(
                height: 14,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(999),
                  gradient: const LinearGradient(
                    colors: [Color(0xFF198754), Color(0xFFFFC107), Color(0xFFDC3545)],
                  ),
                ),
              ),
              Positioned(
                left: (threshold.clamp(0, 1)) * (MediaQuery.of(context).size.width - 60),
                top: -4,
                child: Container(width: 2, height: 22, color: Colors.black87),
              ),
            ],
          ),
          Text('Ngưỡng: ${(threshold * 100).toStringAsFixed(0)}%',
              style: const TextStyle(fontSize: 11, color: Colors.grey)),
          if (riskTier != null) ...[
            const SizedBox(height: 10),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: label == 1 ? const Color(0xFFF8D7DA) : const Color(0xFFD1E7DD),
                borderRadius: BorderRadius.circular(999),
              ),
              child: Text(
                tierLabels[riskTier] ?? riskTier,
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                  color: label == 1 ? const Color(0xFF842029) : const Color(0xFF0F5132),
                ),
              ),
            ),
          ],
          const SizedBox(height: 12),
          ...knowledge.map((g) => _buildKnowledgeGroup(g as Map<String, dynamic>)),
          if (knowledgeError != null)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Text(
                'Ghi chú: $knowledgeError',
                style: const TextStyle(fontSize: 11, color: Colors.grey, fontStyle: FontStyle.italic),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildKnowledgeGroup(Map<String, dynamic> group) {
    final items = (group['items'] as List?) ?? [];
    return Padding(
      padding: const EdgeInsets.only(top: 10),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(group['title']?.toString() ?? '',
              style: const TextStyle(fontSize: 13, fontWeight: FontWeight.bold, color: kPrimary)),
          const SizedBox(height: 6),
          ...items.map((it) {
            final item = it as Map<String, dynamic>;
            return Container(
              margin: const EdgeInsets.only(bottom: 6),
              padding: const EdgeInsets.all(9),
              decoration: BoxDecoration(
                color: const Color(0xFFF6F8FC),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(item['title']?.toString() ?? '',
                      style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold)),
                  if (item['content'] != null)
                    Text(item['content'].toString(), style: const TextStyle(fontSize: 11.5)),
                ],
              ),
            );
          }),
        ],
      ),
    );
  }
}
