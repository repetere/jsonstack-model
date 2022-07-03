import puppeteer from 'puppeteer';

describe('End to End HTML Tests', function(){
  let browser:puppeteer.Browser;
  let page:puppeteer.Page;
  beforeAll(async () => {
    browser = await puppeteer.launch({ headless: true });
    page = await browser.newPage();
  },10000);
  afterAll(async () => {
    await browser.close();
  });
  
  describe('Functional Component Test',()=>{
    it('should render the page', async()=>{
      await page.goto(`file://${__dirname}/test/mock/html/example.html`,{
        waitUntil: 'networkidle2',
      });
      const initialPageData = await page.evaluate(()=>{
        const titleText = document.querySelector('title')?.innerHTML
        return {titleText}
      })
      expect(initialPageData.titleText).toBe('Data test')
      // await page.focus('[name="email"]')
      // await page.keyboard.type('adding from jest')
      // await page.$eval('#formSubmitButton',(el:any)=>el.click())
      // await page.focus('#formResult')
      // await page.keyboard.press('Enter');
      await page.waitForTimeout(10000)
      const loadedPageModelDataCSV1 = await page.evaluate(()=>{
        const csvtest1 = document.querySelector('#csvtest1')?.textContent
        //@ts-ignore
        return {csvtest1}
      })
      expect((loadedPageModelDataCSV1.csvtest1 as string)).toMatch(/answers/gi)
      
      // // await page.screenshot({ path: 'example.png' });
    },30000);
  });
})