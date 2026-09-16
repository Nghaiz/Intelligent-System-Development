import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const HousePriceMobileApp());
}

class HousePriceMobileApp extends StatelessWidget {
  const HousePriceMobileApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Dự đoán giá nhà',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF0D9488)),
        useMaterial3: true,
      ),
      home: const HousePricePredictionPage(),
    );
  }
}

class HousePricePredictionPage extends StatefulWidget {
  const HousePricePredictionPage({super.key});

  @override
  State<HousePricePredictionPage> createState() => _HousePricePredictionPageState();
}

class _HousePricePredictionPageState extends State<HousePricePredictionPage> {
  final _formKey = GlobalKey<FormState>();
  final _areaController = TextEditingController();
  final _frontageController = TextEditingController();
  final _accessRoadController = TextEditingController();
  final _floorsController = TextEditingController();
  final _bedroomsController = TextEditingController();
  final _bathroomsController = TextEditingController();

  // Trình giả lập Android: 10.0.2.2 trỏ tới localhost của máy chủ.
  // Với điện thoại thật, thay bằng địa chỉ IP LAN của máy tính.
  final _apiBaseUrlController = TextEditingController(text: 'http://10.0.2.2:5002');

  static const List<String> _directionOptions = [
    'Bắc', 'Không rõ', 'Nam', 'Tây', 'Tây - Bắc', 'Tây - Nam', 'Đông', 'Đông - Bắc', 'Đông - Nam',
  ];
  static const List<String> _legalOptions = ['Have certificate', 'Không rõ', 'Sale contract'];
  static const List<String> _furnitureOptions = ['Basic', 'Full', 'Không rõ'];
  static const List<String> _provinceOptions = [
    'Hà Nội', 'Hồ Chí Minh', 'Đà Nẵng', 'Hải Phòng', 'Bình Dương', 'Đồng Nai', 'Khác',
  ];
  static const Map<String, String> _modelLabels = {
    'gradient_boosting_regressor': 'Gradient Boosting Regressor (khuyến nghị)',
    'random_forest_regressor': 'Random Forest Regressor',
    'linear_regression': 'Linear Regression',
    'ridge_regression': 'Ridge Regression',
    'decision_tree_regressor': 'Decision Tree Regressor',
  };

  String _houseDirection = 'Không rõ';
  String _balconyDirection = 'Không rõ';
  String _legalStatus = 'Không rõ';
  String _furnitureState = 'Full';
  String _province = 'Hà Nội';
  String _selectedModel = 'gradient_boosting_regressor';

  bool _isLoading = false;
  double? _predictedPrice;
  double? _priceLow;
  double? _priceHigh;
  String? _unit;
  String? _modelLabel;
  String? _interpretation;
  String? _error;

  @override
  void dispose() {
    _areaController.dispose();
    _frontageController.dispose();
    _accessRoadController.dispose();
    _floorsController.dispose();
    _bedroomsController.dispose();
    _bathroomsController.dispose();
    _apiBaseUrlController.dispose();
    super.dispose();
  }

  Future<void> _predict() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() {
      _isLoading = true;
      _error = null;
      _predictedPrice = null;
    });

    final baseUrl = _apiBaseUrlController.text.trim().replaceFirst(RegExp(r'/$'), '');
    final payload = {
      'Area': double.parse(_areaController.text),
      'Frontage': _frontageController.text.isEmpty ? null : double.parse(_frontageController.text),
      'Access Road': _accessRoadController.text.isEmpty ? null : double.parse(_accessRoadController.text),
      'Floors': _floorsController.text.isEmpty ? null : double.parse(_floorsController.text),
      'Bedrooms': _bedroomsController.text.isEmpty ? null : double.parse(_bedroomsController.text),
      'Bathrooms': _bathroomsController.text.isEmpty ? null : double.parse(_bathroomsController.text),
      'House direction': _houseDirection,
      'Balcony direction': _balconyDirection,
      'Legal status': _legalStatus,
      'Furniture state': _furnitureState,
      'Province': _province,
      'model': _selectedModel,
    };

    try {
      final response = await http.post(
        Uri.parse('$baseUrl/house-price/v1/predict'),
        headers: const {'Content-Type': 'application/json'},
        body: jsonEncode(payload),
      );
      final body = jsonDecode(response.body) as Map<String, dynamic>;

      if (response.statusCode != 200) {
        throw Exception(body['error'] ?? 'Không thể dự đoán.');
      }

      setState(() {
        _predictedPrice = (body['predicted_price'] as num).toDouble();
        _priceLow = (body['price_low'] as num).toDouble();
        _priceHigh = (body['price_high'] as num).toDouble();
        _unit = body['unit'] as String;
        _modelLabel = body['model_label'] as String;
        _interpretation = body['interpretation'] as String?;
      });
    } catch (error) {
      setState(() => _error = 'Không thể kết nối API: $error');
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  String? _requiredNumber(String? value) {
    final number = double.tryParse(value ?? '');
    if (number == null || number <= 0) return 'Hãy nhập một số dương hợp lệ.';
    return null;
  }

  String? _optionalNumber(String? value) {
    if (value == null || value.isEmpty) return null;
    final number = double.tryParse(value);
    if (number == null || number < 0) return 'Hãy nhập một số hợp lệ.';
    return null;
  }

  Widget _numberField(String label, TextEditingController controller, {bool required = false}) {
    return TextFormField(
      controller: controller,
      keyboardType: const TextInputType.numberWithOptions(decimal: true),
      validator: required ? _requiredNumber : _optionalNumber,
      decoration: InputDecoration(labelText: label, border: const OutlineInputBorder()),
    );
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
    return Scaffold(
      appBar: AppBar(
        title: const Text('Dự đoán giá nhà'),
        backgroundColor: Theme.of(context).colorScheme.primaryContainer,
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            const Text(
              'Nhập thông tin bất động sản. Kết quả chỉ mang tính tham khảo.',
              style: TextStyle(color: Colors.black54),
            ),
            const SizedBox(height: 20),
            Form(
              key: _formKey,
              child: Column(
                children: [
                  _numberField('Diện tích Area (m²)', _areaController, required: true),
                  const SizedBox(height: 14),
                  _numberField('Mặt tiền Frontage (m)', _frontageController),
                  const SizedBox(height: 14),
                  _numberField('Đường vào Access Road (m)', _accessRoadController),
                  const SizedBox(height: 14),
                  _numberField('Số tầng Floors', _floorsController),
                  const SizedBox(height: 14),
                  _numberField('Số phòng ngủ Bedrooms', _bedroomsController),
                  const SizedBox(height: 14),
                  _numberField('Số phòng tắm Bathrooms', _bathroomsController),
                  const SizedBox(height: 14),
                  _dropdown('Hướng nhà', _houseDirection, _directionOptions, (v) => setState(() => _houseDirection = v!)),
                  const SizedBox(height: 14),
                  _dropdown('Hướng ban công', _balconyDirection, _directionOptions, (v) => setState(() => _balconyDirection = v!)),
                  const SizedBox(height: 14),
                  _dropdown('Tình trạng pháp lý', _legalStatus, _legalOptions, (v) => setState(() => _legalStatus = v!)),
                  const SizedBox(height: 14),
                  _dropdown('Tình trạng nội thất', _furnitureState, _furnitureOptions, (v) => setState(() => _furnitureState = v!)),
                  const SizedBox(height: 14),
                  _dropdown('Tỉnh/Thành phố', _province, _provinceOptions, (v) => setState(() => _province = v!)),
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
                            : const Text('Dự đoán giá nhà'),
                      ),
                    ),
                  ),
                ],
              ),
            ),
            if (_predictedPrice != null) ...[
              const SizedBox(height: 22),
              Card(
                color: Colors.teal.shade50,
                child: Padding(
                  padding: const EdgeInsets.all(18),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('$_predictedPrice $_unit', style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 6),
                      Text('Khoảng dao động: $_priceLow – $_priceHigh $_unit'),
                      const SizedBox(height: 8),
                      Text('Mô hình: $_modelLabel'),
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
                    hintText: 'http://10.0.2.2:5002',
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
