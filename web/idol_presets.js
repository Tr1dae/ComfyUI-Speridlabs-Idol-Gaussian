import { app } from "../../scripts/app.js";

// Keeps the steps / num_gaussians widgets of "Idol Image to 3D Gaussians" in sync with its preset dropdown.
// The preset table comes from the node definition (nodes.py PRESETS), so it is defined in one place only.
app.registerExtension({
    name: "ComfyUI-Idol.presets",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "IdolGenerate") return;
        const presets = nodeData.input?.required?.preset?.[1]?.idol_presets;
        if (!presets) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const widget = (name) => this.widgets?.find((w) => w.name === name);
            const preset = widget("preset");
            if (!preset) return result;

            let applying = false;
            const apply = (name) => {
                const values = presets[name];
                if (!values) return;
                applying = true;
                try {
                    for (const [key, value] of Object.entries(values)) {
                        const target = widget(key);
                        if (target) target.value = value;
                    }
                } finally {
                    applying = false;
                }
                this.setDirtyCanvas?.(true, true);
            };

            const presetCallback = preset.callback;
            preset.callback = function (value, ...rest) {
                const r = presetCallback?.apply(this, [value, ...rest]);
                apply(preset.value);
                return r;
            };

            const controlled = new Set(Object.values(presets).flatMap((values) => Object.keys(values)));
            for (const key of controlled) {
                const target = widget(key);
                if (!target) continue;
                const callback = target.callback;
                target.callback = function (value, ...rest) {
                    const r = callback?.apply(this, [value, ...rest]);
                    const active = presets[preset.value];
                    if (!applying && active && active[key] !== target.value) preset.value = "custom";
                    return r;
                };
            }

            apply(preset.value);
            return result;
        };
    },
});
