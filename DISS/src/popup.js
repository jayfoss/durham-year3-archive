let responseData = null;
let cpc_all = null;
let cpc_advertising = null;
let cpc_analytics = null;
let cpc_functional = null;
let cpc_location = null;
let cpc_device = null;
let cpc_storage = null;
let cpc_personalised = null;
let cpc_social = null;
let cpc_research = null;
let cpc_products = null;
let cpc_offline = null;
let cpc_security = null;
let cpc_other = null;
let all_cpc_checkboxes = [];
let all_cpc_checkboxes_obj = {};

document.addEventListener('DOMContentLoaded', () => {
    window.addEventListener('DOMContentLoaded', () => {
		cpc_all = document.getElementById('cpc-all');
		cpc_advertising = document.getElementById('cpc-advertising');
		cpc_analytics = document.getElementById('cpc-analytics');
		cpc_functional = document.getElementById('cpc-functional');
		cpc_location = document.getElementById('cpc-location');
		cpc_device = document.getElementById('cpc-device');
		cpc_storage = document.getElementById('cpc-storage');
		cpc_personalised = document.getElementById('cpc-personalised');
		cpc_social = document.getElementById('cpc-social');
		cpc_research = document.getElementById('cpc-research');
		cpc_products = document.getElementById('cpc-products');
		cpc_offline = document.getElementById('cpc-offline');
		cpc_security = document.getElementById('cpc-security');
		cpc_other = document.getElementById('cpc-other');
		all_cpc_checkboxes = [
			cpc_advertising,
			cpc_analytics,
			cpc_functional,
			cpc_location,
			cpc_device,
			cpc_storage,
			cpc_personalised,
			cpc_social,
			cpc_research,
			cpc_products,
			cpc_offline,
			cpc_security,
			cpc_other
		];
		all_cpc_checkboxes_obj = {
			'cpc_advertising': cpc_advertising,
			'cpc_analytics': cpc_analytics,
			'cpc_functional': cpc_functional,
			'cpc_location': cpc_location,
			'cpc_device': cpc_device,
			'cpc_storage': cpc_storage,
			'cpc_personalised': cpc_personalised,
			'cpc_social': cpc_social,
			'cpc_research': cpc_research,
			'cpc_products': cpc_products,
			'cpc_offline': cpc_offline,
			'cpc_security': cpc_security,
			'cpc_other': cpc_other
		};
		
		cpc_all.addEventListener('change', (e) => {
			let checkedSettings = {'_action': 'check'};
			for(const check of all_cpc_checkboxes) {
				const inner = check.id.split('-')[1];
				checkedSettings[inner] = e.target.checked;
				check.checked = e.target.checked;
			}
			sendAllCheckedSettings(checkedSettings);
		});
		for(const check of all_cpc_checkboxes) {
			check.addEventListener('change', (e) => {
				sendCheckedSettings(e.target);
			});
		}
		document.getElementById('save-cookie-selection').addEventListener('click', (e) => {
			const checkedSettings = {};
			for(const check of all_cpc_checkboxes) {
				const inner = check.id.split('-')[1];
				checkedSettings[inner] = check.checked;
			}
			checkedSettings.all = cpc_all.checked;
			savePreferences(checkedSettings);
		});
		document.getElementById('apply-cookie-selection').addEventListener('click', (e) => {
			chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
				chrome.tabs.sendMessage(tabs[0].id, {'_action': 'confirm'}, (result) => {
					if (window.chrome.runtime.lastError) {
						console.log(window.chrome.runtime.lastError);
					}
				});
			});
		});
		document.getElementById('rescan-vendors').addEventListener('click', (e) => {
			chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
				chrome.tabs.sendMessage(tabs[0].id, {'_action': 'vendor_rescan'}, (result) => {
					if (window.chrome.runtime.lastError) {
						console.log(window.chrome.runtime.lastError);
					}
				});
			});
		});
		document.getElementById('store-vendors').addEventListener('click', (e) => {
			chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
				chrome.tabs.sendMessage(tabs[0].id, {'_action': 'vendor_store'}, (result) => {
					if (window.chrome.runtime.lastError) {
						console.log(window.chrome.runtime.lastError);
					}
				});
			});
		});
		document.getElementById('apply-vendors').addEventListener('click', (e) => {
			chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
				chrome.tabs.sendMessage(tabs[0].id, {'_action': 'vendor_check'}, (result) => {
					if (window.chrome.runtime.lastError) {
						console.log(window.chrome.runtime.lastError);
					}
				});
			});
		});
		const riskLevelRows = [];
		for(let i = 0; i < 6; i++) {
			riskLevelRows[i] = document.getElementById('risk-' + i);
		}
		document.getElementById('scan-fields').addEventListener('click', (e) => {
			const localDocument = document;
			chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
				chrome.tabs.sendMessage(tabs[0].id, {'_action': 'scan_fields'}, (result) => {
					if (window.chrome.runtime.lastError) {
						console.log(window.chrome.runtime.lastError);
					}
					if(result && result.byLevel) {
						for(const [k, v] of Object.entries(result.byLevel)) {
							riskLevelRows[k].innerHTML = v;
						}
					}
				});
			});
		});
		document.getElementById('should-auto').addEventListener('change', (e) => {
			chrome.storage.sync.set({'should_auto': e.target.checked}, () => {
				console.log(e.target.checked);
			});
		});
		chrome.storage.sync.get(['should_auto'], (auto) => {
			if(!auto) return;
			document.getElementById('should-auto').checked = auto.should_auto;
		});
		chrome.storage.sync.get(['cookie_choices'], (result) => {
			if(!result || !result.cookie_choices) return;
			for(const [k, v] of Object.entries(result.cookie_choices)) {
				if(k === 'all') continue;
				all_cpc_checkboxes_obj['cpc_' + k].checked = v;
			}
			cpc_all.checked = result.cookie_choices.all;
			const obj = Object.assign({}, result.cookie_choices);
			obj._action = 'check';
			sendAllCheckedSettings(obj);
		});
    });
});

function savePreferences(cookieChoices) {
	chrome.storage.sync.set({'cookie_choices': cookieChoices}, () => {
		console.log(cookieChoices);
	});
}

function sendAllCheckedSettings(settings) {
	chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
		chrome.tabs.sendMessage(tabs[0].id, settings, (result) => {
			if (window.chrome.runtime.lastError) {
				console.log(window.chrome.runtime.lastError);
			}
		});
	});
}

function sendCheckedSettings(field) {
	let checkedSettings = {'_action': 'check'};
	checkedSettings[field.id.split('-')[1]] = field.checked;
	chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
		chrome.tabs.sendMessage(tabs[0].id, checkedSettings, (result) => {
			if (window.chrome.runtime.lastError) {
				console.log(window.chrome.runtime.lastError);
			}
		});
	});
}