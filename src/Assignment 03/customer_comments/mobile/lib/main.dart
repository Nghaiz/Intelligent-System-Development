// Ứng dụng Flutter — Phân loại nhận xét khách hàng (Hệ thống 3)
// Gọi REST API thuần NumPy chạy tại http://10.0.2.2:5003/comments/v1
// (10.0.2.2 là địa chỉ máy chủ 127.0.0.1 nhìn từ trình giả lập Android)

import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const CommentsApp());
}

const String kBaseUrl = 'http://10.0.2.2:5003/comments/v1';
const Color kPrimary = Color(0xFF6F42C1);
const Color kAccent = Color(0xFF20C997);

class CommentsApp extends StatelessWidget {
  const CommentsApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Phân loại nhận xét',
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
  final _textCtrl = TextEditingController(text: 'Love this dress, the fabric is so soft!');

  bool _loading = false;
  String? _errorMessage;
  Map<String, dynamic>? _result;
  Map<String, dynamic>? _health;

  static const Map<String, String> tierLabels = {
    'cmt_positive': 'Tích cực',
    'cmt_neutral': 'Trung tính',
    'cmt_negative': 'Tiêu cực',
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
      _textCtrl.text = 'Love this dress, the fabric is so soft!';
    });
  }

  Future<void> _predict() async {
    final text = _textCtrl.text.trim();
    if (text.isEmpty) {
      setState(() => _errorMessage = 'Vui lòng nhập nội dung nhận xét.');
      return;
    }
    setState(() {
      _loading = true;
      _errorMessage = null;
      _result = null;
    });

    try {
      final resp = await http.post(
        Uri.parse('$kBaseUrl/predict'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'text': text}),
      );
      final body = jsonDecode(resp.body) as Map<String, dynamic>;
      if (resp.statusCode == 200) {
        setState(() => _result = body);
      } else {
        setState(() => _errorMessage = body['error']?.toString() ?? 'Đã có lỗi xảy ra.');
      }
    } catch (e) {
      setState(() => _errorMessage = 'Không thể kết nối tới máy chủ (cổng 5003).');
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Phân loại nhận xét'),
        backgroundColor: kPrimary,
        foregroundColor: Colors.white,
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            const Text(
              'Hệ thống 3 · 23,486 đánh giá khách hàng',
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
    return Column(
      children: [
        TextField(
          controller: _textCtrl,
          maxLines: 5,
          decoration: const InputDecoration(
            labelText: 'Nội dung nhận xét',
            border: OutlineInputBorder(),
            hintText: 'Nhập nhận xét của khách hàng…',
          ),
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
    final int label = data['label'] as int;
    final String labelText = data['label_text'] as String;
    final double prob = (data['probability'] as num).toDouble();
    final double threshold = (data['threshold'] as num).toDouble();
    final String? sentimentTier = data['sentiment_tier'] as String?;
    final List matchedTerms = (data['matched_terms'] as List?) ?? [];
    final Map<String, dynamic> tfidf = (data['tfidf'] as Map<String, dynamic>?) ?? {};
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
            labelText,
            style: TextStyle(
              fontSize: 17,
              fontWeight: FontWeight.w800,
              color: label == 1 ? const Color(0xFF198754) : const Color(0xFFDC3545),
            ),
          ),
          const SizedBox(height: 4),
          Text(
            'Xác suất khuyến nghị: ${(prob * 100).toStringAsFixed(1)}% (ngưỡng ${(threshold * 100).toStringAsFixed(0)}%)',
            style: const TextStyle(fontSize: 12, color: Colors.grey),
          ),
          if (sentimentTier != null) ...[
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: const Color(0xFFFFF3CD),
                borderRadius: BorderRadius.circular(999),
              ),
              child: Text(
                tierLabels[sentimentTier] ?? sentimentTier,
                style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Color(0xFF664D03)),
              ),
            ),
          ],
          const SizedBox(height: 12),
          const Text('Từ khoá ảnh hưởng nhiều nhất', style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold)),
          const SizedBox(height: 6),
          Wrap(
            spacing: 6,
            runSpacing: 6,
            children: matchedTerms.map((t) {
              final term = t as Map<String, dynamic>;
              final bool positive = term['polarity'] == 'tích cực';
              return Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                decoration: BoxDecoration(
                  color: positive ? const Color(0xFFD1E7DD) : const Color(0xFFF8D7DA),
                  borderRadius: BorderRadius.circular(999),
                ),
                child: Text(
                  '${term['term']} ${term['weight']}',
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.bold,
                    color: positive ? const Color(0xFF0F5132) : const Color(0xFF842029),
                  ),
                ),
              );
            }).toList(),
          ),
          const SizedBox(height: 10),
          Text(
            'TF-IDF: ${tfidf['n_active_dims']}/${tfidf['total_dims']} chiều (thưa ${tfidf['sparsity_pct']}%)',
            style: const TextStyle(fontSize: 11.5, color: Colors.grey),
          ),
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
                color: const Color(0xFFF6F2FC),
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
