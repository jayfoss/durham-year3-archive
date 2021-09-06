const fieldMap = {}; //Why? For security since we need to escape values used as field ids for XSS prevention.
document.addEventListener('DOMContentLoaded', () => {
    window.addEventListener('DOMContentLoaded', () => {
		chrome.storage.local.get(['vendor_choices'], (result) => {
			const vendorOptions = document.getElementById('vendor-options');
			const choices = result.vendor_choices;
			let allSelected = true;
			for(const [k, v] of Object.entries(choices)) {
				const escaped = e(k);
				const str = '<input type="checkbox" class="custom-control-input" id="' + escaped + '"><label class="custom-control-label" for="' + escaped + '">' + escaped + '</label>';
				const div = document.createElement('div');
				div.className = 'custom-control custom-switch my-2';
				div.innerHTML = str;
				vendorOptions.append(div);
				const newTag = document.getElementById(escaped);
				if(newTag !== null) {
					newTag.checked = v;
					if(v === false) {
						allSelected = false;
					}
				}
				fieldMap[k] = e(k);
			}
			if(allSelected === true) {
				document.getElementById('select-all-vendors').checked = true;
			}
		});
		document.getElementById('save-vendor-selection').addEventListener('click', (e) => {
			const newVendorOptions = Object.assign({}, fieldMap);
			for(const [k, v] of Object.entries(fieldMap)) {
				const t = document.getElementById(v);
				if(t === null) continue;
				newVendorOptions[k] = t.checked;
			}
			chrome.storage.local.set({'vendor_choices': newVendorOptions}, () => {
				console.log('Updated');
			});
		});
		document.getElementById('select-all-vendors').addEventListener('change', (e) => {
			const vendorOptions = document.querySelectorAll('#vendor-options input');
			for(const o of vendorOptions) {
				o.checked = e.target.checked;
			}
		});
	});
});

function e(unsafe) {
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
 }