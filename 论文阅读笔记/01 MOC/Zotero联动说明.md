---
type: moc
tags:
  - moc
  - zotero
created: 2026-07-06
updated: 2026-07-06
---

# Zotero联动说明

## 现在已经完成的部分

- Zotero 已安装
- Better BibTeX 已安装
- Obsidian 论文 vault 已配置
- Zotero Integration / Dataview / Templater 已落到 vault
- Zotero 导入模板已创建：[[07 Templates/Zotero Literature Note Import|Zotero Literature Note Import]]

## 当前 citekey 规则

- `auth.lower + shorttitle(3, 3) + year`

这意味着 citekey 大致会长成：

- `smithtoolsta2025`
- `dongetage2026`

## 建议的 Obsidian Zotero Integration 设置

- Template file:
  `07 Templates/Zotero Literature Note Import.md`
- Output path:
  `02 Papers/{{title}}`
- Image output path:
  `08 Attachments/{{citekey}}`
- Image base name:
  `{{citekey}}`

## 建议的使用方式

1. 在 Zotero 里选中一篇论文
2. 在 Obsidian 里运行 Zotero Integration 的 import 命令
3. 导入后的正式笔记进入 `02 Papers`
4. 批注和图片跟随模板进入同一篇笔记中
5. 后续人工整理后，再在 `topics` 和 `status` 上补全 metadata

## 你需要在 Zotero 里再确认的一项

- `编辑 -> 首选项 -> 导出 -> Quick Copy`

建议选一个 BibTeX / Better BibTeX 兼容的 citation/bibliography 导出方式，确保 Obsidian 的 citation / bibliography 命令都能正常工作。

