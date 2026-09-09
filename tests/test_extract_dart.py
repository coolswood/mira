"""Dart symbol extraction — classes/mixins/enums/extensions, qualified methods.

Enables indexing for Flutter/Dart repositories: typed signatures need a
dedicated extractor (the generic JS/C-like path can't match
`Widget build(BuildContext context)`), and top-level functions are common
in Dart, unlike Java.
"""

from __future__ import annotations

from mira.index.extract import extract_symbols, find_symbol_by_name

FLUTTER_STYLE = """\
import 'package:flutter/material.dart';
import 'package:psy/models/mood_entry.dart';

/// Mood chart widget used by the home screen.
class MoodChart extends StatelessWidget {
  final List<MoodEntry> entries;
  final ValueChanged<MoodEntry>? onEntryTap;

  const MoodChart({super.key, required this.entries, this.onEntryTap});

  @override
  Widget build(BuildContext context) {
    if (entries.isEmpty) {
      return const SizedBox.shrink();
    }
    return _buildChart(context);
  }

  Widget _buildChart(BuildContext context) {
    return CustomPaint(
      painter: MoodPainter(entries: entries),
      child: GestureDetector(
        onTap: () => _handleTap(context),
      ),
    );
  }

  void _handleTap(BuildContext context) {
    showModalBottomSheet(context: context, builder: (_) => const MoodSheet());
  }
}

abstract class ChartDataSource {
  List<MoodEntry> load({required DateTime from});

  bool get isReady;
}

mixin ChartAnimation on StatelessWidget {
  void animate() {
    print('animating');
  }
}

enum ChartPeriod { day, week, month }

extension MoodListX on List<MoodEntry> {
  MoodEntry? get latest {
    return isEmpty ? null : last;
  }

  List<MoodEntry> since(DateTime from) {
    return where((e) => e.createdAt.isAfter(from)).toList();
  }
}

MoodEntry? pickEntry(List<MoodEntry> entries, int index) {
  return index < entries.length ? entries[index] : null;
}

Future<void> main() async {
  runApp(const MoodApp());
}
"""


def test_dart_classes_are_extracted():
    symbols = extract_symbols(FLUTTER_STYLE, "dart")
    classes = {s.name for s in symbols if s.kind == "class"}
    assert {"MoodChart", "ChartDataSource", "ChartAnimation", "ChartPeriod", "MoodListX"} <= classes


def test_dart_methods_get_qualified_names():
    symbols = extract_symbols(FLUTTER_STYLE, "dart")
    by_qual = {s.qualified_name: s for s in symbols if s.qualified_name}
    assert by_qual["MoodChart.build"].kind == "method"
    assert "CustomPaint" in by_qual["MoodChart._buildChart"].source
    assert by_qual["MoodListX.since"].name == "since"
    assert "MoodEntry? pickEntry" not in by_qual


def test_dart_top_level_functions():
    symbols = extract_symbols(FLUTTER_STYLE, "dart")
    functions = {s.name: s for s in symbols if s.kind == "function"}
    assert "pickEntry" in functions
    assert "main" in functions
    # functions have no enclosing type → no qualified name
    assert not functions["main"].qualified_name


def test_dart_constructor_is_matched():
    symbols = extract_symbols(FLUTTER_STYLE, "dart")
    by_qual = {s.qualified_name: s for s in symbols if s.qualified_name}
    assert "MoodChart.MoodChart" in by_qual


def test_dart_statements_are_not_symbols():
    symbols = extract_symbols(FLUTTER_STYLE, "dart")
    names = {s.name for s in symbols}
    # control flow inside method bodies must not false-match
    assert "if" not in names
    assert "print" not in names
    assert "where" not in names
    assert "showModalBottomSheet" not in names


def test_find_symbol_by_qualified_name():
    span = find_symbol_by_name(FLUTTER_STYLE, "dart", "MoodChart._handleTap")
    assert span is not None
    assert span.name == "_handleTap"
    assert find_symbol_by_name(FLUTTER_STYLE, "dart", "build") is not None


NAMED_CTORS_AND_GETTERS = """\
class MoodChart {
  final List<int> entries;

  factory MoodChart.fromJson(Map<String, dynamic> json) {
    return MoodChart(entries: json['entries']);
  }

  factory MoodChart.demo() => MoodChart(entries: []);

  MoodChart(this.entries);

  bool get ready {
    if (entries.isEmpty) {
      return false;
    }
    return true;
  }
}
"""


def test_dart_named_constructors_keep_their_name():
    symbols = extract_symbols(NAMED_CTORS_AND_GETTERS, "dart")
    by_qual = {s.qualified_name: s for s in symbols if s.qualified_name}
    assert by_qual["MoodChart.fromJson"].name == "fromJson"
    assert by_qual["MoodChart.demo"].name == "demo"
    # unnamed constructor is preserved as Class.Class
    assert by_qual["MoodChart.MoodChart"].name == "MoodChart"
    assert find_symbol_by_name(NAMED_CTORS_AND_GETTERS, "dart", "MoodChart.fromJson") is not None


def test_dart_block_bodied_getter_is_extracted_not_scanned():
    symbols = extract_symbols(NAMED_CTORS_AND_GETTERS, "dart")
    by_qual = {s.qualified_name: s for s in symbols if s.qualified_name}
    assert by_qual["MoodChart.ready"].kind == "method"
    # the getter body must not leak control flow as symbols
    names = {s.name for s in symbols}
    assert "if" not in names
    assert "return" not in names
