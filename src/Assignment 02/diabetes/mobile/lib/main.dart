import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const DiabetesMobileApp());
}

class DiabetesMobileApp extends StatelessWidget {
  const DiabetesMobileApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Dự đoán tiểu đường',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF2563EB)),
        useMaterial3: true,
      ),
      home: const DiabetesPredictionPage(),
    );
  }
}

class DiabetesPredictionPage extends StatefulWidget {
  const DiabetesPredictionPage({super.key});

  @override
  State<DiabetesPredictionPage> createState() => _DiabetesPredictionPageState();
}

class _DiabetesPredictionPageState extends State<DiabetesPredictionPage> {
  final _formKey = GlobalKey<FormState>();
  final _glucoseController = TextEditingController();
  final _bmiController = TextEditingController();
  final _ageController = TextEditingController();
  final _pregnanciesController = TextEditingController();
  final _pedigreeController = TextEditingController();

  // Trình giả lập Android: 10.0.2.2 trỏ tới localhost của máy chủ.
  // Với điện thoại thật, thay bằng địa chỉ IP LAN của máy tính.
  final _apiBaseUrlController = TextEditingController(text: 'http://10.0.2.2:5001');

  static const Map<String, String> _modelLabels = {
    'random_forest': 'Random Forest (khuyến nghị)',
    'logistic_regression': 'Logistic Regression',
    'knn': 'K-Nearest Neighbors',
    'decision_tree': 'Decision Tree',
    'svm_rbf': 'SVM (RBF)',
  };

  String _selectedModel = 'random_forest';
  bool _isLoading = false;
  String? _prediction;
  String? _modelLabel;
  double? _confidence;
  String? _interpretation;
  bool? _isHighRisk;
  String? _error;

  @override
  void dispose() {
    _glucoseController.dispose();
    _bmiController.dispose();
    _ageController.dispose();
    _pregnanciesController.dispose();
    _pedigreeController.dispose();
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
      'Glucose': double.parse(_glucoseController.text),
      'BMI': double.parse(_bmiController.text),
      'Age': double.parse(_ageController.text),
      'Pregnancies': double.parse(_pregnanciesController.text),
      'DiabetesPedigreeFunction': double.parse(_pedigreeController.text),
      'model': _selectedModel,
    };

    try {
      final response = await http.post(
        Uri.parse('$baseUrl/diabetes/v1/predict'),
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
        _interpretation = body['interpretation'] as String?;
        _isHighRisk = body['risk_level'] == 'high';
      });
    } catch (error) {
      setState(() => _error = 'Không thể kết nối API: $error');
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  String? _validateNumber(String? value, {bool allowZero = false}) {
    final number = double.tryParse(value ?? '');
    if (number == null) return 'Hãy nhập một số hợp lệ.';
    if (allowZero ? number < 0 : number <= 0) return 'Giá trị không hợp lệ.';
    return null;
  }

  Widget _numberField(String label, TextEditingController controller, {bool allowZero = false}) {
    return TextFormField(
      controller: controller,
      keyboardType: const TextInputType.numberWithOptions(decimal: true),
      validator: (value) => _validateNumber(value, allowZero: allowZero),
      decoration: InputDecoration(labelText: label, border: const OutlineInputBorder()),
    );
  }

  @override
  Widget build(BuildContext context) {
    final resultColor = _isHighRisk == true ? Colors.red.shade700 : Colors.green.shade700;
    final resultBackground = _isHighRisk == true ? Colors.red.shade50 : Colors.green.shade50;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Dự đoán nguy cơ tiểu đường'),
        backgroundColor: Theme.of(context).colorScheme.primaryContainer,
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            const Text(
              'Kết quả chỉ hỗ trợ tham khảo, không thay thế chẩn đoán của bác sĩ.',
              style: TextStyle(color: Colors.black54),
            ),
            const SizedBox(height: 20),
            Form(
              key: _formKey,
              child: Column(
                children: [
                  _numberField('Glucose', _glucoseController),
                  const SizedBox(height: 14),
                  _numberField('BMI', _bmiController),
                  const SizedBox(height: 14),
                  _numberField('Tuổi', _ageController),
                  const SizedBox(height: 14),
                  _numberField('Số lần mang thai', _pregnanciesController, allowZero: true),
                  const SizedBox(height: 14),
                  _numberField('Diabetes Pedigree Function', _pedigreeController, allowZero: true),
                  const SizedBox(height: 14),
                  DropdownButtonFormField<String>(
                    initialValue: _selectedModel,
                    decoration: const InputDecoration(labelText: 'Mô hình', border: OutlineInputBorder()),
                    items: _modelLabels.entries
                        .map((entry) => DropdownMenuItem(value: entry.key, child: Text(entry.value)))
                        .toList(),
                    onChanged: (value) => setState(() => _selectedModel = value!),
                  ),
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
                      Text('Độ tin cậy của mô hình: ${_confidence?.toStringAsFixed(2) ?? 'N/A'}%'),
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
                    hintText: 'http://10.0.2.2:5001',
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
