(() => {
  const $ = id => document.getElementById(id);
  const storeKey = 'roger-chat-ui-v3';
  let chats = [], activeId = null, settings = {textSize:'15px',enterSend:true,reduceMotion:false,showTimestamps:false,temperature:0.7,topP:0.9,maxTokens:80};
  let generator=null, modelPromise=null, modelDevice='webgpu';
  let toastTimer;
  const makeId = () => 'c' + Date.now().toString(36) + Math.random().toString(36).slice(2,7);
  const freshChat = () => ({id:makeId(),title:'New chat',messages:[],created:Date.now(),draft:'',pinned:false});
  function save(){try{localStorage.setItem(storeKey,JSON.stringify({chats,activeId,settings}));}catch(e){}}
  function load(){try{const d=JSON.parse(localStorage.getItem(storeKey)||'null');if(d&&Array.isArray(d.chats)){chats=d.chats;activeId=d.activeId;settings={...settings,...d.settings};settings.temperature=Number(Math.max(0.1,Math.min(1.2,Number(settings.temperature)||0.7)).toFixed(1));settings.topP=Number(Math.max(0.5,Math.min(1,Number(settings.topP)||0.9)).toFixed(2));settings.maxTokens=Math.round(Math.max(1,Math.min(80,Number(settings.maxTokens)||80)));}}catch(e){}if(!chats.length){const c=freshChat();chats=[c];activeId=c.id;}if(!chats.some(c=>c.id===activeId))activeId=chats[0].id;chats.forEach(c=>{if(typeof c.draft!=='string')c.draft='';if(typeof c.pinned!=='boolean')c.pinned=false;});}
  function active(){return chats.find(c=>c.id===activeId)||chats[0];}
  function showToast(text){$('toast').textContent=text;$('toast').classList.add('show');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').classList.remove('show'),1900);}
  function safeText(text){return String(text??'');}
  function renderList(){
    const list=$('chatList');const existingIds=new Set([...list.querySelectorAll('[data-chat-id]')].map(el=>el.dataset.chatId));list.replaceChildren();
    const query=$('chatSearch').value.trim().toLowerCase();
    const visibleChats=chats.filter(c=>!query||(c.title+' '+c.messages.map(m=>m.text||'').join(' ')).toLowerCase().includes(query));
    visibleChats.slice().sort((a,b)=>Number(!!b.pinned)-Number(!!a.pinned)||b.created-a.created).forEach(c=>{
      const row=document.createElement('div');row.className='chat-item'+(c.id===activeId?' active':'');row.setAttribute('data-chat-id',c.id);row.style.animationDelay=Math.min(chats.indexOf(c)*25,150)+'ms';if(existingIds.has(c.id))row.style.animation='none';
      const open=document.createElement('button');open.type='button';open.className='chat-item';open.style.cssText='display:flex;flex:1;min-width:0;padding:0;border:0;min-height:30px;background:transparent;color:inherit;text-align:left';open.setAttribute('aria-label','Open '+c.title);
      const title=document.createElement('span');title.className='chat-title';title.textContent=(c.pinned?'★ ':'')+(c.title||'Untitled chat');
      open.append(title);open.addEventListener('click',()=>{activeId=c.id;render();closeSidebar();});
      const del=document.createElement('button');del.type='button';del.className='delete-chat';del.textContent='×';del.title='Delete chat';del.setAttribute('aria-label','Delete '+(c.title||'chat'));del.addEventListener('click',e=>{e.stopPropagation();deleteChat(c.id);});
      row.append(open,del);list.append(row);
    });
  }
  function renderMessages(){
    const c=active();const existingMessageIds=new Set([...$('messages').querySelectorAll('[data-message-id]')].map(el=>el.dataset.messageId));$('currentTitle').textContent=c.title||'Untitled chat';$('messages').replaceChildren();
    const hasMessages=c.messages.length>0;$('welcome').hidden=hasMessages;
    c.messages.forEach(m=>{
      const row=document.createElement('div');row.className='message-row '+(m.role==='assistant'?'assistant':'user');row.dataset.messageId=m.id||'';if(m.id&&existingMessageIds.has(m.id))row.style.animation='none';
      if(m.role==='assistant'){const avatar=document.createElement('div');avatar.className='message-avatar';avatar.textContent='R!';row.append(avatar);}
      const bubble=document.createElement('div');bubble.className='message';
      if(m.typing){const dots=document.createElement('div');dots.className='typing-dots';dots.setAttribute('aria-label','Roger is typing');for(let i=0;i<3;i++)dots.append(document.createElement('span'));bubble.append(dots);}
      else{bubble.textContent=safeText(m.text);if(m.streaming){const caret=document.createElement('span');caret.className='typing-caret';bubble.append(caret);}}
      const tools=document.createElement('div');tools.className='message-tools';if(!m.typing&&!m.streaming){const action=(label,title,icon,fn)=>{const b=document.createElement('button');b.type='button';b.className='message-action';b.setAttribute('aria-label',title);b.dataset.tooltip=title;b.innerHTML=icon+'<span>'+label+'</span>';b.addEventListener('click',fn);tools.append(b);};const icons={edit:'<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m4 16-.8 4.8L8 20l11-11-4-4zM13.5 6.5l4 4"/></svg>',regenerate:'<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 7v5h-5M4 17v-5h5"/><path d="M5.5 9A7 7 0 0 1 18 6l2 2M4 16l2 2a7 7 0 0 0 12.5-3"/></svg>',fork:'<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="6" cy="5" r="2"/><circle cx="18" cy="19" r="2"/><circle cx="18" cy="5" r="2"/><path d="M6 7v5a5 5 0 0 0 5 5h5m-5-5 5-2"/></svg>',copy:'<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="12" height="13" rx="2"/><path d="M16 8V5a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v11a2 2 0 0 0 2 2h2"/></svg>'};if(m.role==='user')action('Edit','Edit and resend',icons.edit,()=>editMessage(m.id));else action('Regenerate','Regenerate response',icons.regenerate,()=>regenerateMessage(m.id));action('Fork','Branch conversation here',icons.fork,()=>forkFromMessage(m.id));action('Copy','Copy this message',icons.copy,()=>copyMessage(m));if(settings.showTimestamps&&m.created){const t=document.createElement('time');t.className='message-time';t.textContent=new Date(m.created).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});tools.append(t);}bubble.append(tools);}row.append(bubble);$('messages').append(row);
    });
    requestAnimationFrame(()=>{const v=$('chatView');if(v.dataset.stickToBottom!=='false')v.scrollTop=v.scrollHeight;updateJumpButton();});
    $('renameOpen').disabled=false;
  }
  function render(){renderList();renderMessages();applySettings();save();syncDraft();}
  function syncDraft(){const c=active();if($('prompt')&&document.activeElement!==$('prompt'))$('prompt').value=c.draft||'';fitInput();}
  function editMessage(id){if(replyBusy){showToast('Stop the current reply before editing.');return;}const c=active(),i=c.messages.findIndex(m=>m.id===id&&m.role==='user');if(i<0)return;$('prompt').value=c.messages[i].text||'';c.draft=$('prompt').value;c.messages=c.messages.slice(0,i);renderMessages();fitInput();save();$('prompt').focus();showToast('Edit it, then send again.');}
  function regenerateMessage(id){if(replyBusy){showToast('Wait for the current reply to finish.');return;}const c=active(),i=c.messages.findIndex(m=>m.id===id&&m.role==='assistant');if(i<0)return;let u=i-1;while(u>=0&&c.messages[u].role!=='user')u--;if(u<0)return;const prompt=c.messages[u].text;c.messages=c.messages.slice(0,i);const m={id:makeId(),role:'assistant',text:'',typing:true,created:Date.now()};c.messages.push(m);replyBusy=true;const run={cancelled:false,reply:m};activeRun=run;render();$('stop').hidden=false;setTimeout(()=>generateReply(c.id,m,prompt,c.messages.slice(0,-1),run),50);}
  async function copyMessage(m){try{await navigator.clipboard.writeText(String(m.text||''));showToast('Message copied!');}catch(e){showToast('Clipboard permission denied.');}}
  function forkFromMessage(id){if(replyBusy){showToast('Stop the current reply before forking.');return;}const s=active(),i=s.messages.findIndex(m=>m.id===id);if(i<0)return;const d=freshChat();d.title=(s.title||'Chat')+' — branch';d.messages=s.messages.slice(0,i+1).filter(m=>!m.typing&&!m.streaming).map(m=>({...m,id:makeId()}));chats.unshift(d);activeId=d.id;render();showToast('Branched from this point.');}
  function showStats(){const ms=active().messages.filter(m=>!m.typing&&!m.streaming),u=ms.filter(m=>m.role==='user'),a=ms.filter(m=>m.role==='assistant'),words=ms.reduce((n,m)=>n+(String(m.text||'').trim().match(/\S+/g)||[]).length,0),chars=ms.reduce((n,m)=>n+String(m.text||'').length,0),avg=a.length?Math.round(a.reduce((n,m)=>n+String(m.text||'').length,0)/a.length):0;$('statsContent').replaceChildren();[['Messages',ms.length],['Your prompts',u.length],['Roger responses',a.length],['Total words',words.toLocaleString()],['Total characters',chars.toLocaleString()],['Average response',avg+' characters']].forEach(([k,v])=>{const row=document.createElement('div');row.className='stat-row';const l=document.createElement('span');l.textContent=k;const val=document.createElement('strong');val.textContent=v;row.append(l,val);$('statsContent').append(row);});openModal('statsModal');}
  function duplicateChat(){const s=active(),d={...s,id:makeId(),title:(s.title||'Chat')+' (copy)',created:Date.now(),messages:s.messages.map(m=>({...m,id:makeId()})),draft:s.draft||'',pinned:false};chats.unshift(d);activeId=d.id;render();showToast('Chat duplicated.');}
  async function copyLastResponse(){const m=active().messages.slice().reverse().find(x=>x.role==='assistant'&&!x.typing&&!x.streaming&&String(x.text||'').trim());if(m)await copyMessage(m);else showToast('No completed response yet.');}
  function updateJumpButton(){const v=$('chatView');$('jumpLatest').classList.toggle('visible',v.scrollHeight-v.scrollTop-v.clientHeight>180);}
  function createChat(){const c=freshChat();chats.unshift(c);activeId=c.id;render();$('chatView').classList.remove('chat-enter');void $('chatView').offsetWidth;$('chatView').classList.add('chat-enter');setTimeout(()=>$('chatView').classList.remove('chat-enter'),400);$('prompt').value='';fitInput();$('prompt').focus();closeSidebar();}
  function deleteChat(id){
    if(!chats.some(x=>x.id===id))return;
    const row=$('chatList').querySelector('[data-chat-id="'+id+'"]');
    if(row&&!settings.reduceMotion){row.classList.add('removing');setTimeout(()=>finishDelete(id),190);}else finishDelete(id);
  }
  function finishDelete(id){
    if(!chats.some(x=>x.id===id))return;
    if(chats.length===1){chats=[freshChat()];activeId=chats[0].id;render();showToast('Chat deleted.');return;}
    chats=chats.filter(x=>x.id!==id);if(activeId===id)activeId=chats[0].id;render();showToast('Chat deleted.');
  }
  let replyBusy=false,legacyNoticeShown=false,activeRun=null;
  async function loadModel(){
    if(modelPromise)return modelPromise;
    modelPromise=(async()=>{
      try{
        const {RogerSparkBrowser}=await import('./browser_model.js');
        generator=await new RogerSparkBrowser().initialize();
        generator.legacyMode=false;
        return generator;
      }catch(webgpuError){
        // Keep the site usable while the new GPU checkpoint has not been exported.
        const response=await fetch('./roger_spark_weights.json',{cache:'no-store'});
        if(!response.ok)throw webgpuError;
        const bundle=await response.json();
        const {RogerSparkBrowser:LegacyRogerSparkBrowser}=await import('./browser_model_legacy.js');
        generator=new LegacyRogerSparkBrowser(bundle);
        generator.legacyMode=true;
        return generator;
      }
    })().catch(error=>{modelPromise=null;throw error;});
    return modelPromise;
  }
  async function generateReply(chatId,msg,userText,history,run=activeRun){
    try{
      const model=await loadModel();
      if(model.legacyMode&&!legacyNoticeShown){showToast('Using the previous model until the 25M WebGPU model is exported.');legacyNoticeShown=true;}
      const reply=await model.reply(userText,{
        temperature:settings.temperature,
        topP:settings.topP,
        maxTokens:Math.min(128,Number(settings.maxTokens)||80)
      });
      if(run?.cancelled)return;typeReply(chatId,msg,reply,run);
    }catch(error){
      console.error('Roger Spark browser model failed:',error);
      if(run?.cancelled)return;typeReply(chatId,msg,'Roger Spark could not load its model weights. Refresh the page and try again.',run);
    }
  }
  function typeReply(chatId,msg,fullText,run=activeRun){if(run?.cancelled)return;
    msg.typing=false;msg.streaming=true;msg.text='';save();
    let i=0;
    function tick(){
      if(run?.cancelled)return;
      if(i<fullText.length){
        msg.text+=fullText[i++];const c=chats.find(x=>x.id===chatId);
        if(c&&activeId===chatId){const row=$('messages').querySelector('[data-message-id="'+msg.id+'"]');const bubble=row?.querySelector('.message');if(bubble){bubble.textContent=msg.text;const caret=document.createElement('span');caret.className='typing-caret';bubble.append(caret);$('chatView').scrollTop=$('chatView').scrollHeight;}}
        setTimeout(tick,22+Math.random()*22);
      }else{msg.streaming=false;replyBusy=false;if(activeRun===run)activeRun=null;$('stop').hidden=true;save();if(activeId===chatId)renderMessages();$('send').disabled=!$('prompt').value.trim();}
    }tick();
  }
  function sendMessage(text){
    const t=text.trim();if(!t||replyBusy)return;const c=active();
    c.messages.push({id:makeId(),role:'user',text:t,created:Date.now()});
    if(c.title==='New chat'||c.title==='Untitled chat'){const titleLimit=34;c.title=t.length>titleLimit?t.slice(0,titleLimit-1)+'…':t;}
    c.draft='';const targetId=c.id;replyBusy=true;const run={cancelled:false,reply:null};activeRun=run;render();$('prompt').value='';fitInput();$('send').disabled=true;$('stop').hidden=false;$('prompt').focus();
    const reply={id:makeId(),role:'assistant',text:'',typing:true,created:Date.now()};run.reply=reply;c.messages.push(reply);save();if(activeId===targetId)renderMessages();const history=c.messages.slice(0,-1);setTimeout(()=>generateReply(targetId,reply,t,history,run),100);
  }
  function fitInput(){$('prompt').style.height='auto';$('prompt').style.height=Math.min($('prompt').scrollHeight,160)+'px';$('send').disabled=!$('prompt').value.trim()||replyBusy;$('promptCount').textContent=$('prompt').value.length.toLocaleString()+' characters';const c=active();if(c){c.draft=$('prompt').value;save();}$('draftStatus').textContent=$('prompt').value?'Draft saved for this chat':'Draft saved per chat';}
  function applySettings(){
    document.documentElement.style.setProperty('--font-size',settings.textSize);
    document.body.classList.toggle('reduce-motion',!!settings.reduceMotion);
    $('textSize').value=settings.textSize;$('enterSetting').setAttribute('aria-checked',String(!!settings.enterSend));$('motionSetting').setAttribute('aria-checked',String(!!settings.reduceMotion));$('timestampsSetting').setAttribute('aria-checked',String(!!settings.showTimestamps));
    $('temperatureSetting').value=settings.temperature;$('temperatureValue').value=settings.temperature;
    $('topPSetting').value=settings.topP;$('topPValue').value=settings.topP;
    $('maxTokensSetting').value=String(settings.maxTokens);
  }
  function openModal(id){$(id).classList.add('open');const focus=$(id).querySelector('button,input,select');if(focus)focus.focus();}
  function closeModals(){$$('.overlay').forEach(el=>el.classList.remove('open'));}
  function $$(sel){return [...document.querySelectorAll(sel)];}
  function closeSidebar(){$('sidebar').classList.remove('open');}
  $('copyChat').addEventListener('click',async()=>{
    const c=active();
    const transcript=c.messages
      .filter(m=>!m.typing && !m.streaming && String(m.text??'').trim())
      .map(m=>(m.role==='assistant'?'RogerVGB':'User')+': '+String(m.text).trim())
      .join('\n');
    if(!transcript){showToast('No messages to copy yet.');return;}
    try{
      await navigator.clipboard.writeText(transcript);
      showToast('Chat copied!');
    }catch(error){
      const temp=document.createElement('textarea');
      temp.value=transcript;temp.setAttribute('readonly','');
      temp.style.position='fixed';temp.style.opacity='0';document.body.append(temp);
      temp.select();
      const copied=document.execCommand('copy');
      temp.remove();
      showToast(copied?'Chat copied!':'Could not copy chat.');
    }
  });
  $('sidebarToggle').addEventListener('click',()=>{if(window.matchMedia('(max-width: 760px)').matches){closeSidebar();}else{document.body.classList.toggle('sidebar-collapsed');const collapsed=document.body.classList.contains('sidebar-collapsed');$('sidebarToggle').setAttribute('aria-label',collapsed?'Expand sidebar':'Collapse sidebar');$('sidebarToggle').dataset.tooltip=collapsed?'Expand sidebar':'Collapse sidebar';}});
  $('menuOpen').addEventListener('click',()=>{$('sidebar').classList.toggle('open');});
  $('newChat').addEventListener('click',createChat);
  $('pinChat').addEventListener('click',()=>{const x=active();x.pinned=!x.pinned;$('pinChat').classList.toggle('is-pinned',x.pinned);renderList();save();showToast(x.pinned?'Chat pinned.':'Chat unpinned.');});
  $('duplicateChat').addEventListener('click',duplicateChat);$('copyLast').addEventListener('click',copyLastResponse);$('chatStats').addEventListener('click',showStats);
  $('exportChat').addEventListener('click',()=>{const x=active(),p={title:x.title,created:new Date(x.created).toISOString(),messages:x.messages.filter(m=>!m.typing&&!m.streaming).map(m=>({role:m.role,text:m.text,created:m.created?new Date(m.created).toISOString():null}))};const fmt=(window.prompt('Export as txt or json?','txt')||'').trim().toLowerCase();if(!fmt)return;if(!['txt','json'].includes(fmt)){showToast('Choose txt or json.');return;}const body=fmt==='json'?JSON.stringify(p,null,2):p.messages.map(m=>'['+(m.created||'time unknown')+'] '+(m.role==='assistant'?'Roger Spark':'You')+':\n'+m.text).join('\n\n');const blob=new Blob([body],{type:fmt==='json'?'application/json':'text/plain'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=((x.title||'roger-chat').replace(/[^a-z0-9-_]+/gi,'-').slice(0,50)||'roger-chat')+'.'+fmt;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);showToast('Conversation exported.');});
  $('clearChat').addEventListener('click',()=>openModal('clearModal'));$('clearConfirm').addEventListener('click',()=>{const x=active();if(replyBusy&&activeRun){activeRun.cancelled=true;replyBusy=false;activeRun=null;$('stop').hidden=true;}x.messages=[];render();closeModals();showToast('Messages cleared; chat kept.');});
  $('chatSearch').addEventListener('input',renderList);
  $('composer').addEventListener('submit',e=>{e.preventDefault();sendMessage($('prompt').value);});
  $('prompt').addEventListener('input',fitInput);
  $('timestampsSetting').addEventListener('click',()=>{settings.showTimestamps=!settings.showTimestamps;applySettings();renderMessages();save();});
  $('promptStarter').addEventListener('change',e=>{if(e.target.value){$('prompt').value+=e.target.value;e.target.value='';fitInput();$('prompt').focus();}});
  $('stop').addEventListener('click',()=>{if(!replyBusy)return;if(activeRun){activeRun.cancelled=true;const m=activeRun.reply;if(m){m.typing=false;m.streaming=false;m.text=m.text||'[Generation stopped]';}activeRun=null;}replyBusy=false;$('stop').hidden=true;$('send').disabled=!$('prompt').value.trim();save();renderMessages();showToast('Stopped display; model inference may continue in the background.');});
  $('prompt').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&settings.enterSend){e.preventDefault();$('composer').requestSubmit();}});
  $$('.suggestion').forEach(b=>b.addEventListener('click',()=>{$('prompt').value=b.textContent;fitInput();$('prompt').focus();}));
  $('settingsOpen').addEventListener('click',()=>openModal('settingsModal'));
  $('aboutOpen').addEventListener('click',()=>openModal('aboutModal'));
  $('renameOpen').addEventListener('click',()=>{ $('renameInput').value=active().title;openModal('renameModal');});
  $('renameSave').addEventListener('click',()=>{const t=$('renameInput').value.trim();if(!t){$('renameInput').focus();return;}active().title=t;render();closeModals();showToast('Chat renamed.');});
  $('renameInput').addEventListener('keydown',e=>{if(e.key==='Enter')$('renameSave').click();});
  $('textSize').addEventListener('change',e=>{settings.textSize=e.target.value;applySettings();save();});
  $('enterSetting').addEventListener('click',()=>{settings.enterSend=!settings.enterSend;applySettings();save();});
  $('motionSetting').addEventListener('click',()=>{settings.reduceMotion=!settings.reduceMotion;applySettings();save();});
  const clampSetting=(value,min,max,step)=>Math.round(Math.max(min,Math.min(max,Number(value)))/step)*step;
  function updateTemperature(value,source){
    const n=clampSetting(value,0.1,1.2,0.1);
    settings.temperature=Number(n.toFixed(2));
    $('temperatureSetting').value=settings.temperature;
    $('temperatureValue').value=settings.temperature;
    save();
  }
  function updateTopP(value){
    const n=clampSetting(value,0.5,1,0.05);
    settings.topP=Number(n.toFixed(2));
    $('topPSetting').value=settings.topP;
    $('topPValue').value=settings.topP;
    save();
  }
  function commitMaxTokens(){
    const field=$('maxTokensSetting');
    if(field.value.trim()===''){field.value=String(settings.maxTokens);return;}
    settings.maxTokens=Math.round(Math.max(1,Math.min(80,Number(field.value)||settings.maxTokens)));
    field.value=String(settings.maxTokens);
    save();
  }
  $('temperatureSetting').addEventListener('input',e=>updateTemperature(e.target.value,'slider'));
  $('temperatureValue').addEventListener('change',e=>updateTemperature(e.target.value,'number'));
  $('topPSetting').addEventListener('input',e=>updateTopP(e.target.value));
  $('topPValue').addEventListener('change',e=>updateTopP(e.target.value));
  $('maxTokensSetting').addEventListener('change',commitMaxTokens);
  $('maxTokensSetting').addEventListener('blur',commitMaxTokens);
  $('trainingTest').addEventListener('click',()=>{closeModals();if(replyBusy){showToast('Wait for the current reply to finish.');return;}sendMessage('Is the sun hot?');});
  $('settingsDone').addEventListener('click',()=>{save();closeModals();showToast('Settings saved.');});
  $$('[data-close]').forEach(b=>b.addEventListener('click',closeModals));
  $$('.overlay').forEach(o=>o.addEventListener('click',e=>{if(e.target===o)closeModals();}));
  document.addEventListener('keydown',e=>{
    if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='k'){e.preventDefault();closeModals();$('chatSearch').focus();$('chatSearch').select();}
    if((e.metaKey||e.ctrlKey)&&e.shiftKey&&e.key.toLowerCase()==='o'){e.preventDefault();createChat();}
    if(e.key==='Escape'){if($('chatSearch').value){$('chatSearch').value='';renderList();}closeModals();closeSidebar();}
  });
  $('menuOpen').addEventListener('click',()=>$('sidebar').classList.toggle('open'));
  $('chatView').addEventListener('scroll',()=>{const v=$('chatView');v.dataset.stickToBottom=v.scrollHeight-v.scrollTop-v.clientHeight<80?'true':'false';updateJumpButton();});
  $('jumpLatest').addEventListener('click',()=>{$('chatView').dataset.stickToBottom='true';$('chatView').scrollTo({top:$('chatView').scrollHeight,behavior:settings.reduceMotion?'auto':'smooth'});updateJumpButton();});
  load();render();fitInput();$('pinChat').classList.toggle('is-pinned',active().pinned);
  loadModel().catch(()=>{});
})();