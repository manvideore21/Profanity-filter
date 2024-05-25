<template>
  <nav class="bg-white border-gray-200 dark:bg-gray-900">
     <div class="max-w-screen-xl flex flex-wrap items-center justify-between mx-auto p-4">
       <a href="https://flowbite.com/" class="flex items-center space-x-3 rtl:space-x-reverse">
         <img src="https://flowbite.com/docs/images/logo.svg" class="h-8" alt="Flowbite Logo" />
         <span class="self-center text-2xl font-semibold whitespace-nowrap dark:text-white">Profanity Filter</span>
       </a>
       <button @click="toggleNavbar" type="button" class="inline-flex items-center p-2 w-10 h-10 justify-center text-sm text-gray-500 rounded-lg md:hidden hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-gray-200 dark:text-gray-400 dark:hover:bg-gray-700 dark:focus:ring-gray-600" aria-controls="navbar-default" aria-expanded="false">
         <span class="sr-only">Open main menu</span>
         <svg class="w-5 h-5" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 17 14">
           <path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M1 1h15M1 7h15M1 13h15"/>
         </svg>
       </button>
       <div :class="{ 'hidden': !isNavbarOpen, 'w-full': true, 'md:block': true, 'md:w-auto': true }" id="navbar-default">
         <ul class="font-medium flex flex-col p-4 md:p-0 mt-4 border border-gray-100 rounded-lg bg-gray-50 md:flex-row md:space-x-8 rtl:space-x-reverse md:mt-0 md:border-0 md:bg-white dark:bg-gray-800 md:dark:bg-gray-900 dark:border-gray-700">
           <li>
             <a href="#" class="block py-2 px-3 text-white bg-blue-700 rounded md:bg-transparent md:text-blue-700 md:p-0 dark:text-white md:dark:text-blue-500" aria-current="page">Home</a>
           </li>
           <li>
             <a href="#" class="block py-2 px-3 text-gray-900 rounded hover:bg-gray-100 md:hover:bg-transparent md:border-0 md:hover:text-blue-700 md:p-0 dark:text-white md:dark:hover:text-blue-500 dark:hover:bg-gray-700 dark:hover:text-white md:dark:hover:bg-transparent">About</a>
           </li>
           <li>
             <a href="#" class="block py-2 px-3 text-gray-900 rounded hover:bg-gray-100 md:hover:bg-transparent md:border-0 md:hover:text-blue-700 md:p-0 dark:text-white md:dark:hover:text-blue-500 dark:hover:bg-gray-700 dark:hover:text-white md:dark:hover:bg-transparent">Services</a>
           </li>
         </ul>
       </div>
     </div>
  </nav>
  <div class="container mx-auto px-4 mt-8">
     <h1 class="text-4xl font-bold mb-4">Profanity Filter</h1>
     <form @submit.prevent="checkProfanity" class="space-y-4">
       <div>
         <label for="text-input" class="block text-sm font-medium text-gray-700">Enter text to check for profanity:</label>
         <textarea id="text-input" v-model="text" required class="w-full h-32 p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"></textarea>
       </div>
       <button type="submit" class="w-full py-2 px-4 bg-blue-500 text-white font-semibold rounded-lg hover:bg-blue-700">Check Profanity</button>
     </form>
     <div v-if="result" class="mt-4">
       <p class="text-lg">{{ result }}</p>
     </div>
  </div>
 </template>
 
 <script> 
export default {
 data() {
    return {
      text: '',
      result: '',
      isNavbarOpen: false
    };
 },
 methods: {
    async checkProfanity() {
      try {
        const response = await fetch('http://127.0.0.1:5001/profanity-check', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ text: this.text })
        });

        if (!response.ok) {
          throw new Error('Network response was not ok');
        }

        const data = await response.json();
        // Modify the result message to include the name of the node
        // this.result = `Profanity Checker - ${data.profanity_detected ? 'Profanity detected.' : 'No profanity detected.'}`;
        // this.result = `Model 1 - ${data.model_1_result}`;
        
        this.result = `Result - ${data.result}`;
      } catch (error) {
        console.error('There was a problem with your fetch operation:', error);
        this.result = 'Error checking for profanity.';
      }
    },
    toggleNavbar() {
      this.isNavbarOpen = !this.isNavbarOpen;
    }
 }
};
</script>
<style scoped>
p {
 white-space: pre-line;
}
</style>
