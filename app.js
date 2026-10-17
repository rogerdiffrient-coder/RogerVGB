(() => {
  const $ = id => document.getElementById(id);
  const storeKey = 'roger-chat-ui-v3';
  let chats = [], activeId = null, settings = {textSize:'15px',enterSend:true,reduceMotion:false,temperature:0.7,topP:0.9,maxTokens:80};
  let generator=null, modelPromise=null, modelDevice='webgpu';
  let toastTimer;
  const makeId = () => 'c' + Date.now().toString(36) + Math.random().toString(36).slice(2,7);
  const freshChat = () => ({id:makeId(),title:'New chat',messages:[],created:Date.now()});
  function save(){try{localStorage.setItem(storeKey,JSON.stringify({chats,activeId,settings}));}catch(e){}}
  function load(){try{const d=JSON.parse(localStorage.getItem(storeKey)||'null');if(d&&Array.isArray(d.chats)){chats=d.chats;activeId=d.activeId;settings={...settings,...d.settings};settings.temperature=Number(Math.max(0.1,Math.min(1.2,Number(settings.temperature)||0.7)).toFixed(1));settings.topP=Number(Math.max(0.5,Math.min(1,Number(settings.topP)||0.9)).toFixed(2));settings.maxTokens=Math.round(Math.max(1,Math.min(80,Number(settings.maxTokens)||80)));}}catch(e){}if(!chats.length){const c=freshChat();chats=[c];activeId=c.id;}if(!chats.some(c=>c.id===activeId))activeId=chats[0].id;}
  function active(){return chats.find(c=>c.id===activeId)||chats[0];}
  function showToast(text){$('toast').textContent=text;$('toast').classList.add('show');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').classList.remove('show'),1900);}
  function safeText(text){return String(text??'');}
  function renderList(){
    const list=$('chatList');const existingIds=new Set([...list.querySelectorAll('[data-chat-id]')].map(el=>el.dataset.chatId));list.replaceChildren();
    const query=$('chatSearch').value.trim().toLowerCase();
    const visibleChats=chats.filter(c=>!query||(c.title+' '+c.messages.map(m=>m.text||'').join(' ')).toLowerCase().includes(query));
    visibleChats.slice().sort((a,b)=>b.created-a.created).forEach(c=>{
      const row=document.createElement('div');row.className='chat-item'+(c.id===activeId?' active':'');row.setAttribute('data-chat-id',c.id);row.style.animationDelay=Math.min(chats.indexOf(c)*25,150)+'ms';if(existingIds.has(c.id))row.style.animation='none';
      const open=document.createElement('button');open.type='button';open.className='chat-item';open.style.cssText='display:flex;flex:1;min-width:0;padding:0;border:0;min-height:30px;background:transparent;color:inherit;text-align:left';open.setAttribute('aria-label','Open '+c.title);
      const title=document.createElement('span');title.className='chat-title';title.textContent=c.title||'Untitled chat';
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
      row.append(bubble);$('messages').append(row);
    });
    requestAnimationFrame(()=>$('chatView').scrollTop=$('chatView').scrollHeight);
    $('renameOpen').disabled=false;
  }
  function render(){renderList();renderMessages();applySettings();save();}
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
  let replyBusy=false,legacyNoticeShown=false;
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
  async function generateReply(chatId,msg,userText,history){
    try{
      const model=await loadModel();
      if(model.legacyMode&&!legacyNoticeShown){showToast('Using the previous model until the 25M WebGPU model is exported.');legacyNoticeShown=true;}
      const reply=await model.reply(userText,{
        temperature:settings.temperature,
        topP:settings.topP,
        maxTokens:Math.min(128,Number(settings.maxTokens)||80)
      });
      typeReply(chatId,msg,reply);
    }catch(error){
      console.error('Roger Spark browser model failed:',error);
      typeReply(chatId,msg,'Roger Spark could not load its model weights. Refresh the page and try again.');
    }
  }
  function typeReply(chatId,msg,fullText){
    msg.typing=false;msg.streaming=true;msg.text='';save();
    let i=0;
    function tick(){
      if(i<fullText.length){
        msg.text+=fullText[i++];const c=chats.find(x=>x.id===chatId);
        if(c&&activeId===chatId){const row=$('messages').querySelector('[data-message-id="'+msg.id+'"]');const bubble=row?.querySelector('.message');if(bubble){bubble.textContent=msg.text;const caret=document.createElement('span');caret.className='typing-caret';bubble.append(caret);$('chatView').scrollTop=$('chatView').scrollHeight;}}
        setTimeout(tick,22+Math.random()*22);
      }else{msg.streaming=false;replyBusy=false;save();if(activeId===chatId)renderMessages();$('send').disabled=!$('prompt').value.trim();}
    }tick();
  }
  function sendMessage(text){
    const t=text.trim();if(!t||replyBusy)return;const c=active();
    c.messages.push({id:makeId(),role:'user',text:t,created:Date.now()});
    if(c.title==='New chat'||c.title==='Untitled chat'){const titleLimit=34;c.title=t.length>titleLimit?t.slice(0,titleLimit-1)+'…':t;}
    const targetId=c.id;replyBusy=true;render();$('prompt').value='';fitInput();$('send').disabled=true;$('prompt').focus();
    const reply={id:makeId(),role:'assistant',text:'',typing:true,created:Date.now()};c.messages.push(reply);save();if(activeId===targetId)renderMessages();
    const history=c.messages.slice(0,-1);
    setTimeout(()=>generateReply(targetId,reply,t,history),100);
  }
  function fitInput(){$('prompt').style.height='auto';$('prompt').style.height=Math.min($('prompt').scrollHeight,160)+'px';$('send').disabled=!$('prompt').value.trim();}
  function applySettings(){
    document.documentElement.style.setProperty('--font-size',settings.textSize);
    document.body.classList.toggle('reduce-motion',!!settings.reduceMotion);
    $('textSize').value=settings.textSize;$('enterSetting').setAttribute('aria-checked',String(!!settings.enterSend));$('motionSetting').setAttribute('aria-checked',String(!!settings.reduceMotion));
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
  $('newChat').addEventListener('click',createChat);
  $('chatSearch').addEventListener('input',renderList);
  $('composer').addEventListener('submit',e=>{e.preventDefault();sendMessage($('prompt').value);});
  $('prompt').addEventListener('input',fitInput);
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
  load();render();fitInput();
  loadModel().catch(()=>{});
})();