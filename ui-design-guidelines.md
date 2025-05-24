# UI Design Guidelines & Patterns

## Core Principles

### 1. **Eliminate Duplication**
- Move shared/global settings outside tabs to avoid repetition
- Common settings (model selection, precision, etc.) should be configured once globally
- **Example**: Model configuration moved outside Train/Generate tabs in DreamBooth app

### 2. **Logical Information Hierarchy**
- Global settings → Specific functionality tabs
- Primary controls → Secondary/Advanced controls
- Most important → Least important

### 3. **Visual Balance & Proportions**
- Use column scales to create balanced layouts
- Avoid cramming too many controls in one row (max 3-4)
- Group related controls together with proper spacing

## Layout Patterns

### **Global Configuration Section**
```python
# ✅ Good: Clean, balanced 3-column layout
with gr.Row():
    with gr.Column(scale=2):  # Base model selection
        global_model = gr.Dropdown(
            choices=models,
            label="Base Model",
            allow_custom_value=True,
            interactive=True
        )
    with gr.Column(scale=2):  # Alternative/override
        global_custom_model = gr.Textbox(
            label="Custom Model ID (Optional)",
            placeholder="e.g. runwayml/stable-diffusion-v1-5"
        )
    with gr.Column(scale=3):  # Advanced grouped settings
        with gr.Group():
            gr.Markdown("**Advanced Settings**")
            with gr.Row():
                global_pipeline = gr.Dropdown(
                    choices=list(PIPELINE_MAPPING.keys()),
                    value="Generic",
                    label="Pipeline",
                    interactive=True
                )
                global_precision = gr.Dropdown(
                    choices=PRECISION_OPTIONS,
                    value="f16",
                    label="Precision",
                    interactive=True
                )
            with gr.Row():
                global_scheduler = gr.Dropdown(
                    choices=list(SCHEDULER_MAPPING.keys()),
                    value="UniPC",
                    label="Scheduler",
                    interactive=True
                )
                global_cpu_offload = gr.Checkbox(
                    value=False,
                    label="CPU Offload"
                )
```

### **Tab Layout Patterns**
```python
# ✅ Consistent tab structure with equal height columns
with gr.Row(equal_height=True):
    with gr.Column(scale=1):
        # Primary inputs (grouped)
        with gr.Group():
            primary_input1 = gr.Textbox(...)
            primary_input2 = gr.File(...)

        # Parameters/settings (accordion)
        with gr.Accordion("⚙️ Advanced Settings", open=False):
            advanced_setting1 = gr.Slider(...)
            advanced_setting2 = gr.Dropdown(...)

        # Action button at bottom
        action_button = gr.Button("🚀 Action", variant="primary", size="lg")

    with gr.Column(scale=1):
        # Output/preview at top
        output_gallery = gr.Gallery(...)

        # Status/logs below (grouped)
        with gr.Group():
            status_log = gr.Textbox(lines=8, interactive=True)
            progress = gr.Progress()
```

### **Progressive Disclosure**
- Show essential controls prominently
- Group advanced/technical settings in secondary sections
- Use `gr.Group()` for visual organization, `gr.Accordion()` for collapsible content
- Place outputs/previews above status logs for better visual hierarchy

## Anti-Patterns to Avoid

### ❌ **Layout Mistakes**
- **Cramming**: Putting 5+ controls in one row
- **Unbalanced columns**: Mismatched widths/content
- **Unnecessary rows**: Splitting related controls across multiple rows
- **Awkward spacing**: No logical grouping or visual hierarchy

### ❌ **Information Architecture**
- **Duplicated settings**: Same controls in multiple tabs
- **Scattered configuration**: Related settings in different places
- **Poor grouping**: Unrelated controls grouped together

## Content Guidelines

### **Labels & Descriptions**
- **Clear, concise labels**: "Pipeline" not "Pipeline Type"
- **Helpful info text**: "f16=fast, f32=quality" not verbose explanations
- **Directional accuracy**: Descriptions should match layout flow
- **Optional indicators**: Mark non-required fields as "(Optional)"

### **User Guidance**
- Add emoji to section headers for visual interest: "🎛️ Model Configuration"
- Provide context-appropriate tooltips with `info` parameter
- Use placeholder text effectively: "e.g. runwayml/stable-diffusion-v1-5"
- Remove redundant status headings - let components speak for themselves

## Gradio-Specific Patterns

### **Component Interactivity**
```python
# ✅ Always ensure dropdowns are interactive
dropdown = gr.Dropdown(
    choices=options_list,           # Explicit choices parameter
    value=default_value,           # Set default value
    label="Clear Label",           # Descriptive label
    info="Helpful context",        # Brief tooltip
    interactive=True,              # Explicitly enable interaction
    allow_custom_value=True        # If custom input allowed
)

# ✅ Proper textbox configuration
textbox = gr.Textbox(
    label="Input Label",
    placeholder="Example input...",
    info="What this input does",
    lines=8,                       # Adequate space for content
    max_lines=20,                  # Allow expansion
    interactive=True               # Enable text selection/editing
)
```

### **Effective Component Usage**
- `gr.Group()` for visual grouping without extra spacing
- `gr.Accordion()` for collapsible advanced settings
- `scale` parameter for balanced column layouts
- `info` parameter for concise help text
- `equal_height=True` on rows for consistent column alignment
- Always use `interactive=True` explicitly to avoid disabled components

### **Tab Organization**
- Global settings outside tabs with clear section headers
- Each tab focused on single primary function
- Minimal inter-tab dependencies
- Clear tab purposes with emoji and brief descriptions
- Consistent left-right flow: Inputs → Outputs

## Decision Framework

When designing UI layouts, ask:

1. **Is this setting global or function-specific?**
   - Global → Outside tabs
   - Specific → Inside relevant tab

2. **How many controls am I putting together?**
   - 1-2 → Single row
   - 3-4 → Consider grouping or columns
   - 5+ → Split into logical sections

3. **What's the visual hierarchy?**
   - Most important → Prominent placement
   - Advanced → Secondary grouping
   - Optional → Clear labeling

4. **Does the layout look balanced?**
   - Equal-width columns for equal importance
   - Proportional scaling for different content types
   - Consistent spacing and alignment

## Lessons Learned

### **Key Insights from DreamBooth UI Refactoring**
1. **Global settings eliminate confusion** - Users configure once, use everywhere
2. **Scheduler belongs with model settings** - It's pipeline configuration, not generation parameter
3. **Visual hierarchy matters** - Preview/outputs above logs/status
4. **Interactive=True is crucial** - Gradio components can appear disabled without it
5. **Equal height columns prevent misalignment** - Use `equal_height=True` on rows
6. **Clean is better than cluttered** - Remove redundant status headings

### **Common Pitfalls to Avoid**
- Nested imports inside component creation (causes reactivity issues)
- Missing `choices=` parameter in dropdowns
- Forgetting `interactive=True` on form controls
- Putting too many controls in one row (max 3-4)
- Duplicating settings across tabs
- Poor grouping of related functionality

## Success Metrics

A well-designed UI should:
- ✅ Eliminate user confusion about where to find settings
- ✅ Reduce cognitive load through logical grouping
- ✅ Look intentionally designed, not haphazardly assembled
- ✅ Scale gracefully across different screen sizes
- ✅ Guide users naturally through the workflow
- ✅ Have all interactive components actually working
- ✅ Maintain visual consistency across tabs

Remember: **When in doubt, prioritize user workflow over technical convenience.**
