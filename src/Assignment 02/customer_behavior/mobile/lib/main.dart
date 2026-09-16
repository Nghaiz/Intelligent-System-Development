import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const CustomerBehaviorMobileApp());
}

class CustomerBehaviorMobileApp extends StatelessWidget {
  const CustomerBehaviorMobileApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Dự đoán khuyến nghị khách hàng',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF9333EA)),
        useMaterial3: true,
      ),
      home: const CustomerBehaviorPredictionPage(),
    );
  }
}

class CustomerBehaviorPredictionPage extends StatefulWidget {
  const CustomerBehaviorPredictionPage({super.key});

  @override
  State<CustomerBehaviorPredictionPage> createState() => _CustomerBehaviorPredictionPageState();
}

class _CustomerBehaviorPredictionPageState extends State<CustomerBehaviorPredictionPage> {
  final _formKey = GlobalKey<FormState>();
  final _titleController = TextEditingController();
  final _reviewTextController = TextEditingController();
  final _ageController = TextEditingController();
  final _feedbackCountController = TextEditingController();

  // Trình giả lập Android: 10.0.2.2 trỏ tới localhost của máy chủ.
  // Với điện thoại thật, thay bằng địa chỉ IP LAN của máy tính.
  final _apiBaseUrlController = TextEditingController(text: 'http://10.0.2.2:5003');

  static const List<String> _divisionOptions = ['General', 'General Petite', 'Initmates', 'Unknown'];
  static const List<String> _departmentOptions = ['Bottoms', 'Dresses', 'Intimate', 'Jackets', 'Tops', 'Trend', 'Unknown'];
  static const List<String> _classOptions = ['Dresses', 'Knits', 'Blouses', 'Sweaters', 'Pants', 'Jeans', 'Unknown'];
  static const Map<String, String> _modelLabels = {
    'logistic_regression_hybrid': 'Logistic Regression (bảng + văn bản, khuyến nghị)',
    'linear_svm_hybrid': 'Linear SVM (bảng + văn bản)',
    'multinomial_nb_text': 'Multinomial Naive Bayes (chỉ văn bản)',
    'logistic_regression_tabular': 'Logistic Regression (chỉ bảng)',
    'decision_tree_tabular': 'Decision Tree (chỉ bảng)',
    'random_forest_tabular': 'Random Forest (chỉ bảng)',
  };

  String _divisionName = 'General';
  String _departmentName = 'Dresses';
  String _className = 'Dresses';
  String _selectedModel = 'logistic_regression_hybrid';

  bool _isLoading = false;
  String? _prediction;
  String? _modelLabel;
  double? _confidence;
  String? _representation;
  String? _interpretation;
  bool? _isPositive;
  String? _error;

  @override
  void dispose() {
    _titleController.dispose();
    _reviewTextController.dispose();
    _ageController.dispose();
    _feedbackCountController.dispose();
    _apiBaseUrlController.dispose();
    super.dispose();
  }

  Future<void> _predict() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() {
      _isLoading = true;
      _error = null;
      _prediction = null;
    });

    final baseUrl = _apiBaseUrlController.text.trim().replaceFirst(RegExp(r'/$'), '');
    final payload = {
      'Title': _titleController.text,
      'Review Text': _reviewTextController.text,
      'Age': double.parse(_ageController.text),
      'Positive Feedback Count': _feedbackCountController.text.isEmpty
          ? 0
          : double.parse(_feedbackCountController.text),
      'Division Name': _divisionName,
      'Department Name': _departmentName,
      'Class Name': _className,
      'model': _selectedModel,
    };

    try {
      final response = await http.post(
        Uri.parse('$baseUrl/customer-behavior/v1/predict'),
        headers: const {'Content-Type': 'application/json'},
        body: jsonEncode(payload),
      );
      final body = jsonDecode(response.body) as Map<String, dynamic>;

      if (response.statusCode != 200) {
        throw Exception(body['error'] ?? 'Không thể dự đoán.');
      }

      setState(() {
        _prediction = body['prediction'] as String;
        _modelLabel = body['model_label'] as String;
        _confidence = (body['confidence'] as num?)?.toDouble();
        _representation = body['representation'] as String?;
        _interpretation = body['interpretation'] as String?;
        _isPositive = body['prediction_class'] == 1;
      });
    } catch (error) {
      setState(() => _error = 'Không thể kết nối API: $error');
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  String? _requiredText(String? value) {
    if (value == null || value.trim().isEmpty) return 'Nội dung đánh giá không được để trống.';
    return null;
  }

  String? _requiredAge(String? value) {
    final number = double.tryParse(value ?? '');
    if (number == null || number <= 0 || number > 120) return 'Hãy nhập tuổi hợp lệ (1-120).';
    return null;
  }

  String? _optionalNonNegative(String? value) {
    if (value == null || value.isEmpty) return null;
    final number = double.tryParse(value);
    if (number == null || number < 0) return 'Hãy nhập một số hợp lệ.';
    return null;
  }

  Widget _dropdown(String label, String value, List<String> options, ValueChanged<String?> onChanged) {
    return DropdownButtonFormField<String>(
      initialValue: value,
      decoration: InputDecoration(labelText: label, border: const OutlineInputBorder()),
      items: options.map((opt) => DropdownMenuItem(value: opt, child: Text(opt))).toList(),
      onChanged: onChanged,
    );
  }

  @override
  Widget build(BuildContext context) {
    final resultColor = _isPositive == true ? Colors.green.shade700 : Colors.red.shade700;
    final resultBackground = _isPositive == true ? Colors.green.shade50 : Colors.red.shade50;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Dự đoán khuyến nghị khách hàng'),
        backgroundColor: Theme.of(context).colorScheme.primaryContainer,
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            const Text(
              'Nhập nội dung đánh giá sản phẩm để dự đoán khả năng khách hàng khuyến nghị.',
              style: TextStyle(color: Colors.black54),
            ),
            const SizedBox(height: 20),
            Form(
              key: _formKey,
              child: Column(
                children: [
                  TextFormField(
                    controller: _titleController,
                    decoration: const InputDecoration(labelText: 'Tiêu đề (không bắt buộc)', border: OutlineInputBorder()),
                  ),
                  const SizedBox(height: 14),
                  TextFormField(
                    controller: _reviewTextController,
                    maxLines: 5,
                    validator: _requiredText,
                    decoration: const InputDecoration(labelText: 'Nội dung đánh giá', border: OutlineInputBorder()),
                  ),
                  const SizedBox(height: 14),
                  TextFormField(
                    controller: _ageController,
                    keyboardType: TextInputType.number,
                    validator: _requiredAge,
                    decoration: const InputDecoration(labelText: 'Tuổi khách hàng', border: OutlineInputBorder()),
                  ),
                  const SizedBox(height: 14),
                  TextFormField(
                    controller: _feedbackCountController,
                    keyboardType: TextInputType.number,
                    validator: _optionalNonNegative,
                    decoration: const InputDecoration(labelText: 'Số phản hồi tích cực', border: OutlineInputBorder()),
                  ),
                  const SizedBox(height: 14),
                  _dropdown('Phân khúc', _divisionName, _divisionOptions, (v) => setState(() => _divisionName = v!)),
                  const SizedBox(height: 14),
                  _dropdown('Danh mục', _departmentName, _departmentOptions, (v) => setState(() => _departmentName = v!)),
                  const SizedBox(height: 14),
                  _dropdown('Loại sản phẩm', _className, _classOptions, (v) => setState(() => _className = v!)),
                  const SizedBox(height: 14),
                  _dropdown('Mô hình', _selectedModel, _modelLabels.keys.toList(), (v) => setState(() => _selectedModel = v!)),
                  const SizedBox(height: 20),
                  SizedBox(
                    width: double.infinity,
                    child: FilledButton(
                      onPressed: _isLoading ? null : _predict,
                      child: Padding(
                        padding: const EdgeInsets.all(13),
                        child: _isLoading
                            ? const SizedBox(height: 20, width: 20, child: CircularProgressIndicator())
                            : const Text('Dự đoán'),
                      ),
                    ),
                  ),
                ],
              ),
            ),
            if (_prediction != null) ...[
              const SizedBox(height: 22),
              Card(
                color: resultBackground,
                child: Padding(
                  padding: const EdgeInsets.all(18),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(_prediction!, style: TextStyle(fontSize: 19, fontWeight: FontWeight.bold, color: resultColor)),
                      const SizedBox(height: 8),
                      Text('Mô hình: $_modelLabel'),
                      Text('Độ tin cậy: ${_confidence?.toStringAsFixed(2) ?? 'N/A'}%'),
                      Text('Biểu diễn: $_representation'),
                      if (_interpretation != null) ...[
                        const SizedBox(height: 8),
                        Text(_interpretation!),
                      ],
                    ],
                  ),
                ),
              ),
            ],
            if (_error != null) ...[
              const SizedBox(height: 22),
              Text(_error!, style: TextStyle(color: Colors.red.shade700)),
            ],
            const SizedBox(height: 28),
            ExpansionTile(
              title: const Text('Cấu hình REST API'),
              childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
              children: [
                TextFormField(
                  controller: _apiBaseUrlController,
                  keyboardType: TextInputType.url,
                  decoration: const InputDecoration(
                    labelText: 'API base URL',
                    hintText: 'http://10.0.2.2:5003',
                    border: OutlineInputBorder(),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
