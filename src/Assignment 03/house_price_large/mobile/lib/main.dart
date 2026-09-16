// Ứng dụng Flutter — Định giá bất động sản (Hệ thống 2)
// Gọi REST API thuần NumPy chạy tại http://10.0.2.2:5002/house-price/v1
// (10.0.2.2 là địa chỉ máy chủ 127.0.0.1 nhìn từ trình giả lập Android)

import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const HousePriceApp());
}

const String kBaseUrl = 'http://10.0.2.2:5002/house-price/v1';
const Color kPrimary = Color(0xFF198754);
const Color kAccent = Color(0xFFFD7E14);

class HousePriceApp extends StatelessWidget {
  const HousePriceApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Định giá bất động sản',
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

  final _bedCtrl = TextEditingController(text: '3');
  final _bathCtrl = TextEditingController(text: '2');
  final _sizeCtrl = TextEditingController(text: '1800');
  final _acreCtrl = TextEditingController(text: '0.25');
  final _cityCtrl = TextEditingController(text: 'Houston');
  final _stateCtrl = TextEditingController(text: 'Texas');
  final _zipCtrl = TextEditingController(text: '77002');

  bool _loading = false;
  String? _errorMessage;
  Map<String, dynamic>? _result;
  Map<String, dynamic>? _health;

  static const Map<String, String> segmentLabels = {
    'house_luxury': 'Phân khúc cao cấp',
    'house_mid': 'Phân khúc trung cấp',
    'house_standard': 'Phân khúc bình dân',
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
        setState(() => _health = jsonDecode(resp.body) as Map<String, dynamic>);
      }
    } catch (_) {}
  }

  void _fillSample() {
    setState(() {
      _bedCtrl.text = '3';
      _bathCtrl.text = '2';
      _sizeCtrl.text = '1800';
      _acreCtrl.text = '0.25';
      _cityCtrl.text = 'Houston';
      _stateCtrl.text = 'Texas';
      _zipCtrl.text = '77002';
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
      'bed': double.tryParse(_bedCtrl.text) ?? 0,
      'bath': double.tryParse(_bathCtrl.text) ?? 0,
      'house_size': double.tryParse(_sizeCtrl.text) ?? 0,
      'acre_lot': _acreCtrl.text.isEmpty ? null : double.tryParse(_acreCtrl.text),
      'city': _cityCtrl.text,
      'state': _stateCtrl.text,
      'zip_code': _zipCtrl.text,
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
      setState(() => _errorMessage = 'Không thể kết nối tới máy chủ (cổng 5002).');
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Định giá bất động sản'),
        backgroundColor: kPrimary,
        foregroundColor: Colors.white,
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            const Text(
              'Hệ thống 2 · 150,000 giao dịch bất động sản',
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
          Row(
            children: [
              Expanded(
                child: TextFormField(
                  controller: _bedCtrl,
                  keyboardType: TextInputType.number,
                  decoration: const InputDecoration(labelText: 'Phòng ngủ (1–12)', border: OutlineInputBorder()),
                  validator: (v) {
                    final n = double.tryParse(v ?? '');
                    if (n == null || n < 1 || n > 12) return '1–12';
                    return null;
                  },
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: TextFormField(
                  controller: _bathCtrl,
                  keyboardType: TextInputType.number,
                  decoration: const InputDecoration(labelText: 'Phòng tắm (1–12)', border: OutlineInputBorder()),
                  validator: (v) {
                    final n = double.tryParse(v ?? '');
                    if (n == null || n < 1 || n > 12) return '1–12';
                    return null;
                  },
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          TextFormField(
            controller: _sizeCtrl,
            keyboardType: TextInputType.number,
            decoration: const InputDecoration(labelText: 'Diện tích sàn sqft (300–10,000)', border: OutlineInputBorder()),
            validator: (v) {
              final n = double.tryParse(v ?? '');
              if (n == null || n < 300 || n > 10000) return 'Diện tích phải trong khoảng 300–10,000';
              return null;
            },
          ),
          const SizedBox(height: 10),
          TextFormField(
            controller: _acreCtrl,
            keyboardType: const TextInputType.numberWithOptions(decimal: true),
            decoration: const InputDecoration(labelText: 'Lô đất (acre) — tuỳ chọn', border: OutlineInputBorder()),
          ),
          const SizedBox(height: 10),
          TextFormField(
            controller: _cityCtrl,
            decoration: const InputDecoration(labelText: 'Thành phố — tuỳ chọn', border: OutlineInputBorder()),
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              Expanded(
                child: TextFormField(
                  controller: _stateCtrl,
                  decoration: const InputDecoration(labelText: 'Tiểu bang', border: OutlineInputBorder()),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: TextFormField(
                  controller: _zipCtrl,
                  decoration: const InputDecoration(labelText: 'Mã bưu điện', border: OutlineInputBorder()),
                ),
              ),
            ],
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

  Widget _geoRow(String label, bool matched) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(fontSize: 12.5)),
          Text(
            matched ? '✓ Khớp' : '✗ Không khớp',
            style: TextStyle(
              fontSize: 12.5,
              fontWeight: FontWeight.bold,
              color: matched ? const Color(0xFF146C43) : const Color(0xFFA12622),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildResult(Map<String, dynamic> data) {
    final String priceDisplay = data['price_display'] as String;
    final num pricePerSqft = data['price_per_sqft'] as num;
    final String segment = data['segment'] as String? ?? '';
    final Map<String, dynamic> geo = (data['geo_encoding'] as Map<String, dynamic>?) ?? {};
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
          Text(priceDisplay, style: const TextStyle(fontSize: 30, fontWeight: FontWeight.w800, color: kPrimary)),
          Text('Giá / sqft: \$${pricePerSqft.toStringAsFixed(2)}', style: const TextStyle(fontSize: 13, color: Colors.grey)),
          if (segment.isNotEmpty) ...[
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: const Color(0xFFFFF3CD),
                borderRadius: BorderRadius.circular(999),
              ),
              child: Text(
                segmentLabels[segment] ?? segment,
                style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Color(0xFF664D03)),
              ),
            ),
          ],
          const SizedBox(height: 12),
          const Text('Cấp mã hoá địa lý', style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold)),
          _geoRow('Tiểu bang', geo['state_matched'] == true),
          _geoRow('Zip3', geo['zip3_matched'] == true),
          _geoRow('Mã bưu điện', geo['zip_code_matched'] == true),
          _geoRow('Thành phố', geo['city_matched'] == true),
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
                color: const Color(0xFFF4F8F5),
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
